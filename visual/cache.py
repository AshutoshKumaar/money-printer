from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


class VisualCache:
    """Manages visual asset caching by prompt/query hashing."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, prompt: str) -> str:
        import re
        clean = re.sub(r'[^\w\s]', ' ', prompt.lower())
        words = clean.split()
        
        # English stopwords to ignore for cache hits
        stopwords = {"a", "an", "the", "at", "in", "on", "of", "and", "or", "for", "with", "by", "to", "from", "is", "was", "were", "are"}
        filtered = [w for w in words if w not in stopwords]
        
        # Sort alphabetically to ignore word order
        filtered.sort()
        
        normalized = " ".join(filtered)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Path | None:
        if len(cache_key) != 64 or not all(c in "0123456789abcdef" for c in cache_key):
            cache_key = self.get_cache_key(cache_key)
        path = self.cache_dir / f"{cache_key}.jpg"
        if path.exists() and path.stat().st_size > 0:
            return path
        return None

    def store(self, cache_key: str, source_path: Path) -> Path:
        if len(cache_key) != 64 or not all(c in "0123456789abcdef" for c in cache_key):
            cache_key = self.get_cache_key(cache_key)
        dest_path = self.cache_dir / f"{cache_key}.jpg"
        if source_path.resolve() != dest_path.resolve():
            shutil.copy2(source_path, dest_path)
        return dest_path

