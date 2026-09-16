"""Conservative Arabic lexical normalization; source text stays untouched."""
import unicodedata
import re

VERSION = "arabic-lexical-v1"
_VARIANTS = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ی": "ي", "ک": "ك",
    **{str(i): str(i) for i in range(10)},
    **{chr(0x660 + i): str(i) for i in range(10)},
    **{chr(0x6F0 + i): str(i) for i in range(10)},
})


def normalize_with_offsets(text):
    chars, offsets = [], []
    for position, char in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", char).lower().translate(_VARIANTS):
            if normalized == "ـ" or (unicodedata.category(normalized) == "Mn"
                                      and "ARABIC" in unicodedata.name(normalized, "")):
                continue
            chars.append(normalized)
            offsets.append(position)
    return "".join(chars), offsets


def normalize(text):
    return normalize_with_offsets(text)[0]


def tokens(text):
    return re.findall(r"[^\W_]+", normalize(text), re.UNICODE)


def lexical_text(text):
    """Keep exact normalized words plus conservative Arabic article variants."""
    normalized = normalize(text)
    variants = []
    for word in tokens(text):
        for prefix in ("وال", "بال", "كال", "فال", "لل", "ال"):
            if word.startswith(prefix) and len(word) - len(prefix) >= 3:
                variants.append(word[len(prefix):])
                break
    return normalized + (" " + " ".join(variants) if variants else "")


def is_rtl(text):
    for char in text:
        direction = unicodedata.bidirectional(char)
        if direction in {"R", "AL"}:
            return True
        if direction == "L":
            return False
    return False


def passage_preview(text, query, limit=300):
    """Locate normalized matches, but return original spelling and punctuation."""
    normalized, offsets = normalize_with_offsets(text)
    positions = [normalized.find(token) for token in tokens(query) if len(token) > 1]
    positions = [position for position in positions if position >= 0]
    center = offsets[min(positions)] if positions and offsets else 0
    start = max(0, center - 70)
    return ("…" if start else "") + text[start:start + limit].replace("\n", " ").strip()
