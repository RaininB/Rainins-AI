#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# path: ai_all_in_one.py
"""All-in-one agent: chat + RAG + bulk ingest + search + autosearch + continuous learning
+ crypto price alerts (toast/Discord) + RSS news + creation commands + safe plugins

Requirements (core):
  pip install requests
Optional but recommended:
  pip install sentence-transformers faiss-cpu readability-lxml lxml beautifulsoup4
  pip install fastapi uvicorn feedparser win10toast

Run CLI:
  python ai_all_in_one.py --cli
Run API:
  uvicorn ai_all_in_one:app --reload
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import sqlite3
import textwrap
import threading
import time
import urllib.parse
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --- Mandatory deps ---------------------------------------------------------
import requests  # mandatory; many components rely on it
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# --- Optional deps ----------------------------------------------------------
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    from bs4 import BeautifulSoup  # type: ignore
    from readability import Document  # type: ignore
    HAS_PARSER = True
except Exception:
    HAS_PARSER = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    HAS_EMB = True
except Exception:
    HAS_EMB = False

try:
    import feedparser  # type: ignore
    HAS_FEED = True
except Exception:
    HAS_FEED = False

try:
    from win10toast import ToastNotifier  # type: ignore
    HAS_TOAST = True
except Exception:
    HAS_TOAST = False

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False

# --- Logging ---------------------------------------------------------------
import logging

LOG_LEVEL = os.environ.get("AIO_LOG", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_all_in_one")

# --- HTTP session with retries --------------------------------------------

DEFAULT_UA = "Mozilla/5.0 (X11; Linux x86_64) AI-Agent/1.0 (+local)"


def make_session(timeout: int = 20, retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """Shared session with retry/backoff; consistent UA.
    Why: avoid transient network failures & rate limits.
    """
    sess = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": DEFAULT_UA})
    # Attach a simple default timeout to all requests via wrapper
    orig_get, orig_post = sess.get, sess.post

    def _get(url, *args, **kwargs):  # type: ignore
        kwargs.setdefault("timeout", timeout)
        return orig_get(url, *args, **kwargs)

    def _post(url, *args, **kwargs):  # type: ignore
        kwargs.setdefault("timeout", timeout)
        return orig_post(url, *args, **kwargs)

    sess.get = _get  # type: ignore
    sess.post = _post  # type: ignore
    return sess


HTTP = make_session()

# --- LLM (Ollama local) ----------------------------------------------------


class OllamaLLM:
    def __init__(self, model: str = "llama3.1", host: str = "http://127.0.0.1:11434"):
        self.model = model
        self.host = host.rstrip("/")
        self._log = logging.getLogger("OllamaLLM")

    def available(self) -> bool:
        try:
            r = HTTP.get(f"{self.host}/api/tags")
            if r.status_code != 200:
                return False
            data = r.json()
            # tags API returns list; be permissive on shape
            text = json.dumps(data) if not isinstance(data, str) else data
            return self.model in text
        except Exception as e:
            self._log.debug("available() error: %s", e)
            return False

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.6) -> str:
        payload = {
            "model": self.model,
            "prompt": (f"<<SYS>>\n{system}\n<</SYS>>\n" if system else "") + prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = HTTP.post(f"{self.host}/api/generate", json=payload)
        r.raise_for_status()
        try:
            return str(r.json().get("response", "")).strip()
        except Exception:
            return r.text.strip()


# --- Embeddings ------------------------------------------------------------


class Embedder:
    """Sentence-Transformers wrapper with dynamic dimension detection."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not HAS_EMB:
            raise RuntimeError("Install: pip install sentence-transformers")
        self.model = SentenceTransformer(model_name)
        try:
            self.dim = int(self.model.get_sentence_embedding_dimension())
        except Exception:
            self.dim = 384  # sensible default for common models

    def embed(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


# --- Vector Store ----------------------------------------------------------


class VectorStore:
    """FAISS IP index with JSON metadata. Thread-safe where needed.
    Why: prevent concurrent add/search races and persist metadata.
    """

    def __init__(self, dim: int, path: str = "store.faiss", meta_path: str = "store_meta.json"):
        self.dim = dim
        self.path = path
        self.meta_path = meta_path
        self.meta: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)
            self._load()
        else:
            self.index = None
            self._load_meta_only()

    def _load(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        if os.path.exists(self.path):
            try:
                self.index = faiss.read_index(self.path)
            except Exception:
                logger.warning("FAISS index load failed; starting empty")
                self.index = faiss.IndexFlatIP(self.dim)

    def _load_meta_only(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

    def add(self, embeddings, metadatas: List[Dict[str, Any]]):
        if len(metadatas) != len(embeddings):
            raise ValueError("embeddings and metadatas length mismatch")
        with self._lock:
            if HAS_FAISS and self.index is not None:
                import numpy as np
                vecs = np.asarray(embeddings, dtype="float32")
                if vecs.shape[1] != self.dim:
                    raise ValueError(f"Vector dim {vecs.shape[1]} != index dim {self.dim}")
                self.index.add(vecs)
            self.meta.extend(metadatas)
            self._save()

    def _save(self):
        if HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, self.path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def search(self, query_vec, k: int = 5) -> List[Dict[str, Any]]:
        if not (HAS_FAISS and self.index is not None and self.index.ntotal > 0):
            return []
        import numpy as np
        with self._lock:
            q = np.asarray([query_vec], dtype="float32")
            D, I = self.index.search(q, k)
        hits: List[Dict[str, Any]] = []
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(self.meta):
                m = dict(self.meta[idx])
                m["_score"] = float(score)
                hits.append(m)
        return hits


# --- Memory (SQLite) -------------------------------------------------------


class MemoryDB:
    """SQLite-backed memory: experiences, docs, alerts, notifications."""

    def __init__(self, path: str = "mind.db"):
        self.path = path
        self._init()

    def _init(self):
        with sqlite3.connect(self.path) as conn:
            c = conn.cursor()
            c.execute(
                """CREATE TABLE IF NOT EXISTS experiences(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, importance REAL, context TEXT, text TEXT
            )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS docs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, url TEXT, title TEXT, snippet TEXT
            )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS alerts(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created REAL,
                pair TEXT,
                rule TEXT,
                channel TEXT
            )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS notify(
                key TEXT PRIMARY KEY,
                value TEXT
            )"""
            )
            conn.commit()

    def add_experience(self, text: str, context: str, importance: float) -> None:
        with sqlite3.connect(self.path) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO experiences(ts,importance,context,text) VALUES(?,?,?,?)",
                (time.time(), importance, context, text),
            )
            conn.commit()

    def top_experiences(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.path) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT ts,importance,context,text FROM experiences ORDER BY importance DESC, ts DESC LIMIT ?",
                (limit,),
            )
            rows = c.fetchall()
        return [
            {"ts": r[0], "importance": r[1], "context": r[2], "text": r[3]} for r in rows
        ]

    def add_doc(self, url: str, title: str, snippet: str) -> None:
        with sqlite3.connect(self.path) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO docs(ts,url,title,snippet) VALUES(?,?,?,?)",
                (time.time(), url, title, snippet[:1000]),
            )
            conn.commit()

    def stats(self) -> Dict[str, int]:
        with sqlite3.connect(self.path) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM experiences")
            e = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM docs")
            d = c.fetchone()[0]
        return {"experiences": int(e), "docs": int(d)}

    def add_alert(self, pair: str, rule: str, channel: str) -> None:
        with sqlite3.connect(self.path) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO alerts(created,pair,rule,channel) VALUES(?,?,?,?)",
                (time.time(), pair.upper(), rule.strip(), channel.strip()),
            )
            conn.commit()

    def list_alerts(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.path) as conn:
            c = conn.cursor()
            c.execute("SELECT id,created,pair,rule,channel FROM alerts ORDER BY id DESC")
            rows = c.fetchall()
        return [
            {"id": r[0], "created": r[1], "pair": r[2], "rule": r[3], "channel": r[4]} for r in rows
        ]

    def save_notify(self, key: str, value: str) -> None:
        with sqlite3.connect(self.path) as conn:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO notify(key,value) VALUES(?,?)", (key, value))
            conn.commit()

    def get_notify(self, key: str) -> Optional[str]:
        with sqlite3.connect(self.path) as conn:
            c = conn.cursor()
            c.execute("SELECT value FROM notify WHERE key=?", (key,))
            row = c.fetchone()
        return row[0] if row else None


# --- Web ingestion ----------------------------------------------------------


class WebIngestor:
    UA = DEFAULT_UA

    def __init__(self, embedder: Embedder, store: VectorStore, db: MemoryDB):
        self.embedder = embedder
        self.store = store
        self.db = db
        self._log = logging.getLogger("WebIngestor")

    def _fetch(self, url: str) -> Optional[str]:
        try:
            r = HTTP.get(url, headers={"User-Agent": self.UA})
            if r.status_code != 200:
                return None
            return r.text
        except Exception:
            return None

    def _clean(self, html: str) -> Tuple[str, str]:
        if HAS_PARSER:
            try:
                doc = Document(html)
                title = doc.short_title() or "Untitled"
                content_html = doc.summary()
                soup = BeautifulSoup(content_html, "lxml")
                text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
                return title, text
            except Exception:
                pass
        # Fallback
        try:
            soup = BeautifulSoup(html, "lxml") if HAS_PARSER else BeautifulSoup(html, "html.parser")
        except Exception:
            return "Untitled", re.sub(r"\s+", " ", html)
        title = soup.title.get_text().strip() if soup.title else "Untitled"
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return title, text

    def _chunk(self, text: str, max_words: int = 800) -> List[str]:
        words = text.split()
        if not words:
            return []
        out, cur = [], []
        for w in words:
            cur.append(w)
            if len(cur) >= max_words:
                out.append(" ".join(cur))
                cur = []
        if cur:
            out.append(" ".join(cur))
        return out

    def ingest_url(self, url: str) -> int:
        html = self._fetch(url)
        if not html:
            return 0
        title, text = self._clean(html)
        chunks = self._chunk(text)
        if not chunks:
            return 0
        vecs = self.embedder.embed(chunks)
        metas = [
            {"type": "doc", "url": url, "title": title, "chunk": i, "text": chunks[i]}
            for i in range(len(chunks))
        ]
        self.store.add(vecs, metas)
        self.db.add_doc(url, title, chunks[0][:300])
        return len(chunks)


# --- Web searcher (DuckDuckGo HTML) ----------------------------------------


class WebSearcher:
    UA = DEFAULT_UA

    def __init__(self):
        if not HAS_PARSER:
            raise RuntimeError("Install: pip install readability-lxml lxml beautifulsoup4")

    def search(self, topic: str, n: int = 5) -> List[str]:
        q = urllib.parse.quote(topic)
        url = f"https://duckduckgo.com/html/?q={q}&ia=web"
        try:
            r = HTTP.get(url, headers={"User-Agent": self.UA})
            r.raise_for_status()
        except Exception:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        links: List[str] = []
        seen: set[str] = set()
        for a in soup.find_all("a", href=True):
            text = (a.get_text() or "").strip()
            if not text:
                continue
            href = a["href"]
            cand: Optional[str] = None
            if href.startswith("/l/?"):
                parsed = urllib.parse.urlparse(href)
                qs = urllib.parse.parse_qs(parsed.query)
                if "uddg" in qs:
                    cand = urllib.parse.unquote(qs["uddg"][0])
            else:
                cand = href
            if not (cand and cand.startswith("http")):
                continue
            if "duckduckgo.com" in cand:
                continue
            if cand not in seen:
                links.append(cand)
                seen.add(cand)
            if len(links) >= n:
                break
        return links


# --- Emergent core (slim) --------------------------------------------------


class EmergentCore:
    def __init__(self, db: MemoryDB):
        self.db = db
        self.short_term: deque[Dict[str, Any]] = deque(maxlen=64)

    def learn(self, text: str, context: str, importance: float = 0.6) -> None:
        importance = max(0.0, min(1.0, float(importance)))
        self.db.add_experience(text=text, context=context, importance=importance)
        self.short_term.appendleft({"text": text, "context": context, "importance": importance})

    def recall_snippets(self, query: str, k: int = 3) -> List[str]:
        exps = self.db.top_experiences(limit=50)
        q_terms = set(query.lower().split())
        scored: List[Tuple[float, str]] = []
        for e in exps:
            overlap = sum(1 for w in q_terms if w in e["text"].lower())
            scored.append((overlap + float(e["importance"]), e["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]


# --- Orchestrator ----------------------------------------------------------

SYSTEM_PROMPT = (
    "Answer concisely with high signal. Cite sources from CONTEXT when present. "
    "If insufficient info is available, say so briefly."
)

SUMMARY_PROMPT = (
    "Using the CONTEXT FROM WEB and CONTEXT FROM MEMORY above if present, write a concise, high-signal summary "
    "of the user's topic. Include bullet points and short citations like [Title](URL) where applicable."
)


class AllInOneAgent:
    def __init__(self, llm: OllamaLLM, embedder: Embedder, store: VectorStore, db: MemoryDB):
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.db = db
        self.core = EmergentCore(db)

    def _retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        qv = self.embedder.embed([query])[0]
        return self.store.search(qv, k=k)

    def answer(self, user: str) -> str:
        hits = self._retrieve(user, k=5)
        hit_txts: List[str] = []
        for h in hits:
            src = f"[{h.get('title','')} ]({h.get('url','')})"
            text = h.get("text", "")
            hit_txts.append(f"SOURCE: {src}\nSNIPPET: {text[:800]}")
        mems = self.core.recall_snippets(user, k=3)
        context = ""
        if hit_txts:
            context += "CONTEXT FROM WEB:\n" + "\n\n".join(hit_txts) + "\n\n"
        if mems:
            context += "CONTEXT FROM MEMORY:\n" + "\n\n".join(mems) + "\n\n"
        prompt = f"{context}USER: {user}\nASSISTANT:"
        out = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        imp = 0.6 + 0.1 * min(1.0, len(out) / 800) + (0.1 if hits else 0.0)
        self.core.learn(text=out, context=user, importance=imp)
        return out


# --- Continuous learning ----------------------------------------------------


class ContinualLearner:
    def __init__(self, ingestor: WebIngestor, interval_sec: int = 300):
        self.ingestor = ingestor
        self.interval = max(10, int(interval_sec))
        self.queue: deque[str] = deque()
        self.enabled = False
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._log = logging.getLogger("ContinualLearner")
        # auto-search
        self.searcher = WebSearcher() if HAS_PARSER else None
        self.autosearch_topic: Optional[str] = None
        self.autosearch_n: int = 3
        self.seen_path = "autosrch_seen.json"
        self._seen: set[str] = set()
        self._load_seen()

    def _load_seen(self):
        try:
            if os.path.exists(self.seen_path):
                self._seen = set(json.load(open(self.seen_path, "r", encoding="utf-8")))
        except Exception:
            self._seen = set()

    def _save_seen(self):
        try:
            json.dump(sorted(list(self._seen)), open(self.seen_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    def start(self):
        self.enabled = True
        if self._t is None or not self._t.is_alive():
            self._stop.clear()
            self._t = threading.Thread(target=self._loop, daemon=True, name="ContinualLearner")
            self._t.start()

    def stop(self):
        self.enabled = False
        self._stop.set()

    def status(self) -> str:
        return (
            f"auto={'on' if self.enabled else 'off'}, queued={len(self.queue)}, "
            f"autosearch_topic={self.autosearch_topic}, autosearch_n={self.autosearch_n}"
        )

    def enqueue(self, url: str):
        self.queue.append(url)

    def set_topic(self, topic: str):
        self.autosearch_topic = topic

    def set_n(self, n: int):
        self.autosearch_n = max(1, min(10, int(n)))

    def _auto_search_step(self):
        if not (self.searcher and self.autosearch_topic):
            return
        try:
            found = self.searcher.search(self.autosearch_topic, n=self.autosearch_n)
        except Exception:
            return
        for u in found:
            if u in self._seen:
                continue
            self._seen.add(u)
            self._save_seen()
            try:
                n = self.ingestor.ingest_url(u)
                logger.info("[autosearch] Ingested %s chunks from %s", n, u)
            except Exception as e:
                logger.warning("[autosearch] Ingest failed for %s: %s", u, e)

    def _loop(self):
        while not self._stop.is_set():
            if self.enabled:
                if self.queue:
                    url = self.queue.popleft()
                    try:
                        n = self.ingestor.ingest_url(url)
                        logger.info("[auto] Ingested %s chunks from %s", n, url)
                    except Exception as e:
                        logger.warning("[auto] Ingest failed for %s: %s", url, e)
                else:
                    self._auto_search_step()
            time.sleep(self.interval)


# --- RSS crypto news ingestor ----------------------------------------------


class RSSIngestor:
    FEEDS = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://news.bitcoin.com/feed/",
        "https://www.theblock.co/rss",
    ]

    def __init__(self, ingestor: WebIngestor):
        self.ingestor = ingestor

    def pull(self, n_per_feed: int = 3):
        if not HAS_FEED:
            print("feedparser not installed: pip install feedparser")
            return
        total = 0
        for f in self.FEEDS:
            try:
                feed = feedparser.parse(f)
                for entry in feed.entries[: n_per_feed or 0]:
                    url = entry.link
                    total += self.ingestor.ingest_url(url)
            except Exception:
                pass
        print(f"[rss] total chunks ingested: {total}")


# --- Price watcher + alerts -------------------------------------------------


class PriceWatcher:
    """Poll prices and evaluate alert rules via CoinGecko."""

    def __init__(self, db: MemoryDB, poll_sec: int = 30):
        self.db = db
        self.poll = max(10, int(poll_sec))
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self.toast = ToastNotifier() if HAS_TOAST else None
        self.history: Dict[str, deque[Tuple[float, float]]] = {}
        self._last_trigger: Dict[Tuple[str, str], float] = {}
        self.cooldown_sec = 120
        self._log = logging.getLogger("PriceWatcher")
        # pair -> (coingecko_id, vs_currency)
        self.map: Dict[str, Tuple[str, str]] = {
            "BTCUSD": ("bitcoin", "usd"),
            "ETHUSD": ("ethereum", "usd"),
        }

    def start(self):
        if self._t and self._t.is_alive():
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True, name="PriceWatcher")
        self._t.start()

    def stop(self):
        self._stop.set()

    def _fetch_price(self, pair: str) -> Optional[float]:
        pair_u = pair.upper()
        if pair_u not in self.map:
            return None
        cg_id, vs = self.map[pair_u]
        try:
            r = HTTP.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": cg_id, "vs_currencies": vs},
            )
            if r.status_code != 200:
                return None
            data = r.json()
            val = data.get(cg_id, {}).get(vs)
            return float(val) if val is not None else None
        except Exception:
            return None

    def _get_pct_change(self, pair: str, seconds: int) -> Optional[float]:
        dq = self.history.get(pair.upper())
        if not dq:
            return None
        now = time.time()
        older: Optional[Tuple[float, float]] = None
        latest: Optional[Tuple[float, float]] = dq[-1] if dq else None
        for ts, px in dq:
            # pick the earliest point that is at least `seconds` old
            if now - ts >= seconds:
                older = (ts, px)
                break
        if not older or not latest:
            return None
        old_px, new_px = older[1], latest[1]
        if old_px == 0:
            return None
        return (new_px - old_px) / old_px * 100.0

    def _record_price(self, pair: str, price: float):
        d = self.history.get(pair.upper())
        if d is None:
            d = deque(maxlen=500)
            self.history[pair.upper()] = d
        d.append((time.time(), price))

    def _notify(self, msg: str, channels: str):
        ch = [c.strip().lower() for c in channels.split(",")]
        print(f"[ALERT] {msg}")  # keep console echo for visibility
        if "toast" in ch and self.toast:
            try:
                self.toast.show_toast("Crypto Alert", msg, duration=5, threaded=False)
            except Exception:
                pass
        if "discord" in ch:
            wh = self.db.get_notify("discord_webhook")
            if wh:
                try:
                    HTTP.post(wh, json={"content": msg})
                except Exception:
                    pass

    def _cooldown_ok(self, pair: str, rule: str) -> bool:
        key = (pair.upper(), rule)
        now = time.time()
        last = self._last_trigger.get(key, 0.0)
        if now - last < self.cooldown_sec:
            return False
        self._last_trigger[key] = now
        return True

    def _eval_rule(self, pair: str, price: float, rule: str) -> Optional[str]:
        r = rule.replace(" ", "").lower()
        # price triggers
        m = re.match(r"price(>=|<=|>|<|==)(\d+(?:\.\d+)?)", r)
        if m:
            op, thresh_s = m.groups()
            thresh = float(thresh_s)
            ok = (
                (op == ">" and price > thresh)
                or (op == "<" and price < thresh)
                or (op == ">=" and price >= thresh)
                or (op == "<=" and price <= thresh)
                or (op == "==" and price == thresh)
            )
            if ok and self._cooldown_ok(pair, rule):
                return f"{pair} price {price:.2f} triggered rule {rule}"
            return None
        # percent change triggers
        m = re.match(r"pct_(\d+)(m|h)(>=|<=|>|<)(-?\d+(?:\.\d+)?)", r)
        if m:
            val_s, unit, op, thresh_s = m.groups()
            val = int(val_s)
            thresh = float(thresh_s)
            seconds = val * 60 if unit == "m" else val * 3600
            pct = self._get_pct_change(pair, seconds)
            if pct is None:
                return None
            ok = (
                (op == ">" and pct > thresh)
                or (op == "<" and pct < thresh)
                or (op == ">=" and pct >= thresh)
                or (op == "<=" and pct <= thresh)
            )
            if ok and self._cooldown_ok(pair, rule):
                return f"{pair} {pct:.2f}%/{val}{unit} triggered rule {rule}"
        return None

    def _loop(self):
        watch_pairs = list(self.map.keys())
        while not self._stop.is_set():
            alerts = get_agent().db.list_alerts()
            for pair in watch_pairs:
                price = self._fetch_price(pair)
                if price is None:
                    continue
                self._record_price(pair, price)
                for a in alerts:
                    if a["pair"].upper() != pair:
                        continue
                    msg = self._eval_rule(pair, price, a["rule"])
                    if msg:
                        self._notify(msg, a["channel"])
            time.sleep(self.poll)


# --- File helpers (creation/plugins) ---------------------------------------


def ensure_dirs():
    os.makedirs("files/summaries", exist_ok=True)
    os.makedirs("files/created", exist_ok=True)
    os.makedirs("plugins", exist_ok=True)


def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-_]+", "_", s)[:80].strip("_")


def write_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# --- FastAPI (optional) ----------------------------------------------------

app = FastAPI(title="All-In-One AI") if HAS_FASTAPI else None

if HAS_FASTAPI:

    class ChatIn(BaseModel):
        message: str

    @app.post("/chat")
    def chat_endpoint(inp: ChatIn):
        agent = get_agent()
        reply = agent.answer(inp.message)
        return {"reply": reply, "stats": agent.db.stats()}


# --- Singletons ------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton_agent: Optional[AllInOneAgent] = None
_singleton_ingestor: Optional[WebIngestor] = None
_singleton_auto: Optional[ContinualLearner] = None
_singleton_searcher: Optional[WebSearcher] = None
_singleton_prices: Optional[PriceWatcher] = None
_singleton_rss: Optional[RSSIngestor] = None


def get_agent() -> AllInOneAgent:
    global _singleton_agent, _singleton_ingestor, _singleton_auto
    if _singleton_agent is None:
        with _singleton_lock:
            if _singleton_agent is None:
                llm = OllamaLLM()
                if not llm.available():
                    raise RuntimeError(
                        "Ollama not reachable. Start: `ollama serve` and pull a compatible model."
                    )
                emb = Embedder()
                store = VectorStore(dim=emb.dim)
                db = MemoryDB()
                _singleton_agent = AllInOneAgent(llm, emb, store, db)
                _singleton_ingestor = WebIngestor(emb, store, db)
                _singleton_auto = ContinualLearner(_singleton_ingestor, interval_sec=180)
                ensure_dirs()
    return _singleton_agent  # type: ignore


def get_ingestor() -> WebIngestor:
    get_agent()
    return _singleton_ingestor  # type: ignore


def get_auto() -> ContinualLearner:
    get_agent()
    return _singleton_auto  # type: ignore


def get_searcher() -> WebSearcher:
    global _singleton_searcher
    if _singleton_searcher is None:
        _singleton_searcher = WebSearcher()
    return _singleton_searcher


def get_pricewatcher() -> PriceWatcher:
    global _singleton_prices
    if _singleton_prices is None:
        _singleton_prices = PriceWatcher(get_agent().db, poll_sec=30)
        _singleton_prices.start()
    return _singleton_prices


def get_rss() -> RSSIngestor:
    global _singleton_rss
    if _singleton_rss is None:
        _singleton_rss = RSSIngestor(get_ingestor())
    return _singleton_rss


# --- Help text -------------------------------------------------------------

HELP = """
Commands:
  :help                          Show this help
  :stats                         Show memory/doc counts
  :ingest <url>                  Fetch+embed one URL
  :bulk <url1> <url2> ...        Ingest many URLs (space-separated)
  :learn <url>                   Queue URL for continuous learning
  :auto on|off|status            Toggle/view continuous learning
  :search <topic> [N]            Find & ingest top N links for topic
  :autosearch status             Show current topic/N and auto status
  :autosearch settopic <topic>   Set auto-search topic
  :autosearch setn <N>           Set how many links to ingest each cycle (1–10)
  :summarize <topic>             RAG summary → files/summaries/<ts>_<slug>.md
  :make <filename> <instruction> Create content → files/created/<filename>
  :code <filename.py> <instr>    Create code → files/created/<filename.py>
  :plugin <name> <instruction>   Create plugins/<name>.py with run(text)
  :runplugin <name> <arg>        Import and run plugins/<name>.py:run(arg)
  :alert add <PAIR> <RULE> <CH>  Add alert (e.g., :alert add BTCUSD price>64000 toast)
  :alerts                        List alerts
  :notify discord set <URL>      Save Discord webhook for alerts
  :rss pull                      Ingest top crypto RSS headlines now
  :quit                          Exit
"""


# --- CLI -------------------------------------------------------------------


def cli_chat():
    agent = get_agent()
    ingestor = get_ingestor()
    auto = get_auto()
    print("All-In-One AI ready. Type ':help' for commands.")
    while True:
        try:
            msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not msg:
            continue

        if msg.lower() in (":q", ":quit", ":exit"):
            print("Bye.")
            break
        if msg.lower() == ":help":
            print(HELP)
            continue
        if msg.lower() == ":stats":
            print(agent.db.stats())
            continue

        if msg.lower().startswith(":ingest "):
            url = msg.split(" ", 1)[1].strip()
            try:
                n = ingestor.ingest_url(url)
                print(f"Ingested {n} chunks from {url}")
            except Exception as e:
                print(f"Ingest failed: {e}")
            continue

        if msg.lower().startswith(":bulk "):
            urls = msg.split(" ", 1)[1].strip().split()
            total = 0
            for u in urls:
                try:
                    n_chunks = ingestor.ingest_url(u)
                    total += n_chunks
                    print(f"[bulk] {u} -> {n_chunks} chunks")
                except Exception as e:
                    print(f"[bulk] {u} -> error: {e}")
            print(f"[bulk] total chunks: {total}")
            continue

        if msg.lower().startswith(":learn "):
            url = msg.split(" ", 1)[1].strip()
            auto.enqueue(url)
            print(f"Queued for auto-ingest: {url}")
            continue

        if msg.lower().startswith(":auto"):
            parts = msg.split()
            if len(parts) == 2 and parts[1].lower() == "on":
                auto.start(); print("Auto-learning: ON"); continue
            if len(parts) == 2 and parts[1].lower() == "off":
                auto.stop(); print("Auto-learning: OFF"); continue
            print(auto.status()); continue

        if msg.lower().startswith(":search "):
            rest = msg.split(" ", 1)[1].strip()
            parts = rest.rsplit(" ", 1)
            topic, n = (parts[0], parts[1]) if len(parts) == 2 and parts[1].isdigit() else (rest, "5")
            n = int(n)
            try:
                searcher = get_searcher()
            except Exception as e:
                print(str(e)); continue
            urls = searcher.search(topic, n=n)
            if not urls:
                print("No links found.")
                continue
            print(f"Found {len(urls)} links. Ingesting...")
            total = 0
            for u in urls:
                try:
                    n_chunks = ingestor.ingest_url(u)
                    total += n_chunks
                    print(f"[search] {u} -> {n_chunks} chunks")
                except Exception as e:
                    print(f"[search] {u} -> error: {e}")
            print(f"[search] total chunks: {total}")
            continue

        if msg.lower().startswith(":autosearch"):
            parts = msg.split()
            if len(parts) == 2 and parts[1].lower() == "status":
                print(auto.status()); continue
            if len(parts) >= 3 and parts[1].lower() == "settopic":
                topic = " ".join(parts[2:])
                auto.set_topic(topic); print(f"autosearch topic set: {topic}"); continue
            if len(parts) == 3 and parts[1].lower() == "setn":
                try:
                    auto.set_n(int(parts[2])); print(f"autosearch N set: {int(parts[2])}")
                except Exception:
                    print("Usage: :autosearch setn <int>")
                continue
            if len(parts) == 2 and parts[1].lower() == "on":
                auto.start(); print("Auto-learning: ON"); continue
            if len(parts) == 2 and parts[1].lower() == "off":
                auto.stop(); print("Auto-learning: OFF"); continue
            print("Usage: :autosearch status | settopic <topic> | setn <N> | on | off"); continue

        if msg.lower().startswith(":summarize "):
            topic = msg.split(" ", 1)[1].strip()
            hits = agent._retrieve(topic, k=6)
            hit_txts: List[str] = []
            for h in hits:
                src = f"[{h.get('title','')}]({h.get('url','')})"
                text = h.get("text", "")
                hit_txts.append(f"SOURCE: {src}\nSNIPPET: {text[:800]}")
            mems = agent.core.recall_snippets(topic, k=5)
            context = ""
            if hit_txts:
                context += "CONTEXT FROM WEB:\n" + "\n\n".join(hit_txts) + "\n\n"
            if mems:
                context += "CONTEXT FROM MEMORY:\n" + "\n\n".join(mems) + "\n\n"
            prompt = f"{context}USER: Summarize topic: {topic}\nASSISTANT:"
            out = agent.llm.generate(prompt, system=SUMMARY_PROMPT)
            ts = int(time.time())
            path = f"files/summaries/{ts}_{slugify(topic)}.md"
            write_text(path, out)
            agent.core.learn(out, context=f"summary:{topic}", importance=0.8)
            print(f"Summary saved: {path}")
            continue

        if msg.lower().startswith(":make "):
            rest = msg.split(" ", 1)[1]
            parts = rest.split(" ", 1)
            if len(parts) < 2:
                print("Usage: :make <filename> <instruction>")
                continue
            fname, instr = parts[0], parts[1]
            out = agent.llm.generate(f"Create the following content:\n{instr}\n\nReturn only the content.")
            path = f"files/created/{fname}"
            write_text(path, out)
            print(f"Created: {path}")
            continue

        if msg.lower().startswith(":code "):
            rest = msg.split(" ", 1)[1]
            parts = rest.split(" ", 1)
            if len(parts) < 2:
                print("Usage: :code <filename.py> <instruction>")
                continue
            fname, instr = parts[0], parts[1]
            out = agent.llm.generate(
                "Write a complete, runnable code file as requested.\n"
                f"Instruction: {instr}\n"
                "Return only the code with no explanation."
            )
            path = f"files/created/{fname}"
            write_text(path, out)
            print(f"Code file created: {path}")
            continue

        if msg.lower().startswith(":plugin "):
            rest = msg.split(" ", 1)[1]
            parts = rest.split(" ", 1)
            if len(parts) < 2:
                print("Usage: :plugin <name> <instruction>")
                continue
            name, instr = parts[0], parts[1]
            code = textwrap.dedent(
                f'''
                """
                Auto-generated plugin: {name}
                Instruction: {instr}
                """
                def run(text: str) -> str:
                    # Why: keep plugin sandbox simple and safe.
                    return f"[{name}] processed: " + text
                '''
            ).strip()
            path = f"plugins/{slugify(name)}.py"
            write_text(path, code)
            print(f"Plugin created: {path}")
            continue

        if msg.lower().startswith(":runplugin "):
            rest = msg.split(" ", 1)[1]
            parts = rest.split(" ", 1)
            if len(parts) < 2:
                print("Usage: :runplugin <name> <arg>")
                continue
            name, arg = parts[0], parts[1]
            mod_path = f"plugins/{slugify(name)}.py"
            if not os.path.exists(mod_path):
                print(f"No such plugin: {mod_path}")
                continue
            try:
                spec = importlib.util.spec_from_file_location(name, mod_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError("Invalid plugin spec")
                mod = importlib.util.module_from_spec(spec)  # type: ignore
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "run"):
                    res = mod.run(arg)  # type: ignore
                    print(res)
                else:
                    print("Plugin has no 'run(text)' function.")
            except Exception as e:
                print(f"Plugin error: {e}")
            continue

        if msg.lower().startswith(":alert add "):
            parts = msg.split()
            if len(parts) < 5:
                print("Usage: :alert add <PAIR> <RULE> <CHANNELS>")
                print("Examples: :alert add BTCUSD price>64000 toast")
                print("          :alert add ETHUSD pct_1h<=-5 toast,discord")
                continue
            pair, rule, channels = parts[2], parts[3], " ".join(parts[4:])
            agent.db.add_alert(pair, rule, channels)
            get_pricewatcher()  # ensure watcher is running
            print(f"Alert added for {pair}: {rule} -> {channels}")
            continue

        if msg.lower().startswith(":alerts"):
            rows = agent.db.list_alerts()
            if not rows:
                print("No alerts yet.")
            else:
                for r in rows:
                    ts = dt.datetime.fromtimestamp(r["created"]).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"#{r['id']} {ts} {r['pair']} :: {r['rule']} -> {r['channel']}")
            continue

        if msg.lower().startswith(":notify discord set "):
            wh = msg.split(" ", 3)[3].strip()
            agent.db.save_notify("discord_webhook", wh)
            print("Discord webhook saved.")
            continue

        if msg.lower().startswith(":rss pull"):
            rss = get_rss()
            rss.pull(n_per_feed=3)
            continue

        # normal chat
        try:
            ans = agent.answer(msg)
        except Exception as e:
            ans = f"Error: {e}"
        print(f"\nAI: {ans}")


# --- Main ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run interactive CLI chat")
    args = parser.parse_args()
    if args.cli:
        try:
            cli_chat()
        except RuntimeError as e:
            logger.error(str(e))
    else:
        if HAS_FASTAPI:
            print("API mode: run with 'uvicorn ai_all_in_one:app --reload'")
        else:
            # Fallback to CLI if FastAPI not installed
            cli_chat()
