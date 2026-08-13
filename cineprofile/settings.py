from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import dotenv_values


CONNECTION_KEYS = (
    "TMDB_TOKEN",
    "RADARR_URL",
    "RADARR_API_KEY",
    "RADARR_ROOT_FOLDER",
    "RADARR_QUALITY_PROFILE_ID",
)


def _environment_file(path: str | Path) -> Path:
    target = Path(path)
    if target.exists() and not target.is_file():
        raise ValueError(
            f"{target} doit être un fichier, mais c’est actuellement un dossier. "
            "Renomme ou supprime ce dossier avant d’enregistrer les connexions."
        )
    return target


def _quoted(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )
    return f'"{escaped}"'


def _update_environment_file(
    path: str | Path,
    updates: dict[str, str | None],
) -> None:
    """Apply all updates through one atomic file replacement."""
    target = _environment_file(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = (
        target.read_text(encoding="utf-8").splitlines()
        if target.exists()
        else []
    )
    pending = dict(updates)
    rendered: list[str] = []
    for line in lines:
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        key = match.group(1) if match else None
        if key not in pending:
            rendered.append(line)
            continue
        value = pending.pop(key)
        if value is not None:
            rendered.append(f"{key}={_quoted(value)}")
    for key, value in pending.items():
        if value is not None:
            rendered.append(f"{key}={_quoted(value)}")

    temporary = target.with_name(f"{target.name}.tmp")
    temporary.write_text(
        "\n".join(rendered) + ("\n" if rendered else ""),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def read_connection_settings(path: str | Path) -> dict[str, str]:
    target = _environment_file(path)
    if not target.exists():
        return {key: "" for key in CONNECTION_KEYS}
    values = dotenv_values(target)
    return {key: str(values.get(key) or "") for key in CONNECTION_KEYS}


def save_connection_settings(
    path: str | Path,
    *,
    tmdb_token: str,
    radarr_url: str = "",
    radarr_api_key: str = "",
    radarr_root_folder: str = "",
    radarr_quality_profile_id: str = "",
) -> None:
    clean_tmdb = tmdb_token.strip()
    clean_url = radarr_url.strip().rstrip("/")
    clean_radarr_key = radarr_api_key.strip()
    if not clean_tmdb:
        raise ValueError("Le jeton TMDB est obligatoire.")
    if bool(clean_url) != bool(clean_radarr_key):
        raise ValueError(
            "L’adresse et la clé API Radarr doivent être renseignées ensemble."
        )
    _update_environment_file(
        path,
        {
            "TMDB_TOKEN": clean_tmdb,
            "RADARR_URL": clean_url or None,
            "RADARR_API_KEY": clean_radarr_key or None,
            "RADARR_ROOT_FOLDER": radarr_root_folder.strip() or None,
            "RADARR_QUALITY_PROFILE_ID": (
                radarr_quality_profile_id.strip() or None
            ),
        },
    )


def save_tmdb_token_file(path: str | Path, token: str) -> None:
    value = token.strip()
    if not value:
        raise ValueError("Le jeton TMDB est vide.")
    _update_environment_file(path, {"TMDB_TOKEN": value})


def read_tmdb_token_file(path: str | Path) -> str:
    return read_connection_settings(path)["TMDB_TOKEN"]


def forget_tmdb_token_file(path: str | Path) -> None:
    _update_environment_file(path, {"TMDB_TOKEN": None})


def save_radarr_connection_file(
    path: str | Path,
    url: str,
    api_key: str,
) -> None:
    clean_url = url.strip().rstrip("/")
    clean_key = api_key.strip()
    if not clean_url or not clean_key:
        raise ValueError("L’adresse et la clé API Radarr sont obligatoires.")
    _update_environment_file(
        path,
        {"RADARR_URL": clean_url, "RADARR_API_KEY": clean_key},
    )


def forget_radarr_connection_file(path: str | Path) -> None:
    _update_environment_file(
        path,
        {
            "RADARR_URL": None,
            "RADARR_API_KEY": None,
            "RADARR_ROOT_FOLDER": None,
            "RADARR_QUALITY_PROFILE_ID": None,
        },
    )
