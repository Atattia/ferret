"""Small, explicit search grammar shared by the desktop and CLI."""

from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import os


def escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@dataclass(frozen=True)
class SearchFilters:
    extensions: tuple[str, ...] = ()
    folder: str | None = None

    def sql(self) -> tuple[str, list[str]]:
        """Return bound predicates for the files alias ``f``."""
        clauses, params = [], []
        if self.extensions:
            clauses.append("(" + " OR ".join(
                "lower(f.filename) LIKE ? ESCAPE '\\'" for _ in self.extensions
            ) + ")")
            params.extend("%." + escape_like(ext.lower().lstrip("."))
                          for ext in self.extensions)
        if self.folder:
            folder = str(Path(self.folder).expanduser().absolute()).rstrip("/\\")
            clauses.append("f.path LIKE ? ESCAPE '\\'")
            params.append(escape_like(folder + os.sep) + "%")
        return (" AND " + " AND ".join(clauses) if clauses else ""), params


def parse_query(query: str) -> tuple[str, SearchFilters]:
    """Recognize type:pdf,md and in:"/folder with spaces"; keep other text."""
    # During typing, unmatched quotes are ordinary text, never a search error.
    try:
        tokens = shlex.split(query, posix=True)
    except ValueError:
        return query.strip(), SearchFilters()
    words, extensions = [], []
    folder = None
    for token in tokens:
        if token.lower().startswith("type:") and re.fullmatch(
            r"\.?[\w]+(?:,\.?[\w]+)*", token[5:]
        ):
            extensions.extend(ext.lower().lstrip(".") for ext in token[5:].split(","))
        elif token.lower().startswith("in:") and token[3:]:
            folder = token[3:]
        else:
            words.append(token)
    return " ".join(words), SearchFilters(tuple(dict.fromkeys(extensions)), folder)
