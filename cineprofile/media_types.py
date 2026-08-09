from __future__ import annotations

import re
import unicodedata


def _normalise_media_type(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def is_series_type(value: object) -> bool:
    """Return whether an IMDb/TMDB type describes television episodic media."""

    normalized = _normalise_media_type(value)
    compact = normalized.replace(" ", "")
    return (
        normalized in {"tv", "television", "emission de television"}
        or "series" in normalized
        or "serie" in normalized
        or "episode" in normalized
        or compact in {"tvseries", "tvminiseries", "tvepisode"}
    )


def tmdb_media_type(value: object) -> str:
    return "tv" if is_series_type(value) else "movie"
