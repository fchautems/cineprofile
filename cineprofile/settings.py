from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values, set_key, unset_key


def save_tmdb_token_file(path: str | Path, token: str) -> None:
    value = token.strip()
    if not value:
        raise ValueError("Le jeton TMDB est vide.")
    set_key(str(Path(path)), "TMDB_TOKEN", value, quote_mode="always")


def read_tmdb_token_file(path: str | Path) -> str:
    target = Path(path)
    if not target.exists():
        return ""
    return str(dotenv_values(target).get("TMDB_TOKEN") or "")


def forget_tmdb_token_file(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        unset_key(str(target), "TMDB_TOKEN")

