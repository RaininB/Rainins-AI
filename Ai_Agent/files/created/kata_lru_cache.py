
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value  # move to the end to mark as recently used
            return value
        else:
            return -1

    def put(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)  # remove the first item (least recently used)
        self.cache[key] = value

# Demo
cache = LRUCache(2)  # Create a cache with capacity 2
print(cache.put(1, 1))   # Insert 1 into cache
print(cache.put(2, 2))   # Insert 2 into cache
print(cache.get(1))     # Returns 1 as it was the most recently used item in cache
cache.put(3, 3)         # Inserts 3 into cache. Since this is a LRU Cache, the least recently accessed items are removed.
print(cache.get(2))     # Returns -1 because we have removed 2 from the cache when we inserted 3.
