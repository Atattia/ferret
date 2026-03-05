import hashlib
from pathlib import Path


def hash_file(path: str | Path) -> str:
    """Return SHA256 hex digest of file contents."""
    path = Path(path)
    sha256 = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        print(f"[hasher] Failed to hash {path}: {e}")
        return ""
