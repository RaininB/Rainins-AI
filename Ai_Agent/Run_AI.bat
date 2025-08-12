@echo off
chcp 65001 >nul

REM --- stay out of System32 ---
cd /d "%USERPROFILE%\Documents\AI_Agent"

REM --- unbuffered output so you see progress ---
set PYTHONUNBUFFERED=1

echo.
echo Starting the AI chat. If this is the first run, it may download an embeddings model (can take a few minutes).
echo Make sure Ollama is running (you already pulled llama3.1).
echo.

py -3 -m pip install --quiet --disable-pip-version-check sentence-transformers readability-lxml lxml beautifulsoup4 requests >nul 2>&1

py -3 ai_all_in_one.py --cli

echo.
echo (Window will stay open so you can read any messages.)
pause
