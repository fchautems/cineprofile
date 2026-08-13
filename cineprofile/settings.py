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


def save_radarr_connection_file(
    path: str | Path,
    url: str,
    api_key: str,
) -> None:
    clean_url = url.strip().rstrip("/")
    clean_key = api_key.strip()
    if not clean_url or not clean_key:
        raise ValueError("L’adresse et la clé API Radarr sont obligatoires.")
    target = str(Path(path))
    set_key(target, "RADARR_URL", clean_url, quote_mode="always")
    set_key(target, "RADARR_API_KEY", clean_key, quote_mode="always")


def forget_radarr_connection_file(path: str | Path) -> None:
    target = Path(path)
    if not target.exists():
        return
    unset_key(str(target), "RADARR_URL")
    unset_key(str(target), "RADARR_API_KEY")
