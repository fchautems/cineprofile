from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

import pandas as pd

from .db import connect, initialize, transaction


IMDB_ID_PATTERN = re.compile(r"(tt\d{5,12})", re.IGNORECASE)


ALIASES = {
    "imdb_id": ["Const", "IMDb ID", "imdb_id", "Title ID"],
    "title": ["Title", "Primary Title", "title"],
    "original_title": ["Original Title", "OriginalTitle", "original_title"],
    "title_type": ["Title Type", "TitleType", "Type", "title_type"],
    "year": ["Year", "Release Year", "year"],
    "user_rating": ["Your Rating", "YourRating", "Rating", "user_rating"],
    "date_rated": ["Date Rated", "DateRated", "date_rated"],
    "imdb_rating": ["IMDb Rating", "IMDbRating", "IMDb rating", "imdb_rating"],
    "num_votes": ["Num Votes", "NumVotes", "Number of Votes", "num_votes"],
    "runtime_minutes": [
        "Runtime (mins)",
        "Runtime (Minutes)",
        "Runtime",
        "runtime_minutes",
    ],
    "release_date": ["Release Date", "ReleaseDate", "release_date"],
    "genres_csv": ["Genres", "Genre", "genres_csv"],
    "directors_csv": ["Directors", "Director", "directors_csv"],
    "url": ["URL", "IMDb URL", "url"],
}


@dataclass(frozen=True)
class ImportResult:
    total_rows: int
    imported_rows: int
    updated_rows: int
    skipped_rows: int
    columns: tuple[str, ...]


def _find_column(frame: pd.DataFrame, logical_name: str) -> str | None:
    normalized = {str(column).strip().casefold(): column for column in frame.columns}
    for alias in ALIASES[logical_name]:
        if alias.casefold() in normalized:
            return str(normalized[alias.casefold()])
    return None


def _read_csv(source: str | Path | bytes | BinaryIO | TextIO) -> pd.DataFrame:
    if isinstance(source, io.TextIOBase):
        return pd.read_csv(source, dtype=str, keep_default_na=False)
    if isinstance(source, bytes):
        try:
            return pd.read_csv(
                io.BytesIO(source),
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                io.BytesIO(source),
                dtype=str,
                keep_default_na=False,
                encoding="cp1252",
            )
    try:
        return pd.read_csv(
            source,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(
            source,
            dtype=str,
            keep_default_na=False,
            encoding="cp1252",
        )


def _clean_text(value: object) -> str | None:
    if value is None or bool(pd.isna(value)):
        return None
    text = str(value).strip() if value is not None else ""
    return text or None


def _number(value: object, integer: bool = False) -> float | int | None:
    text = _clean_text(value)
    if text is None:
        return None
    cleaned = (
        text.replace("’", "")
        .replace("'", "")
        .replace("\u202f", "")
        .replace(" ", "")
    )
    cleaned = cleaned.replace(",", "") if integer else cleaned.replace(",", ".")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return int(number) if integer else number


def _imdb_id(row: pd.Series, id_column: str | None, url_column: str | None) -> str | None:
    candidates = []
    if id_column:
        candidates.append(row.get(id_column))
    if url_column:
        candidates.append(row.get(url_column))
    for candidate in candidates:
        match = IMDB_ID_PATTERN.search(str(candidate))
        if match:
            return match.group(1).lower()
    return None


def normalize_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {name: _find_column(frame, name) for name in ALIASES}
    required = ["title", "user_rating"]
    missing = [name for name in required if not columns[name]]
    if missing:
        raise ValueError(
            "Colonnes IMDb introuvables : "
            + ", ".join(missing)
            + ". Utilise l’export CSV de la page « Your Ratings »."
        )

    records: list[dict[str, object]] = []
    for index, row in frame.iterrows():
        imdb_id = _imdb_id(row, columns["imdb_id"], columns["url"])
        # L'identifiant est normalement toujours présent. Ce fallback permet
        # seulement d'analyser d'anciens exports, sans prétendre les enrichir.
        if not imdb_id:
            imdb_id = f"legacy-{index:08d}"

        def value(name: str) -> object:
            column = columns[name]
            return row.get(column) if column else None

        user_rating = _number(value("user_rating"))
        title = _clean_text(value("title"))
        if (
            title is None
            or user_rating is None
            or not 1.0 <= float(user_rating) <= 10.0
        ):
            continue

        records.append(
            {
                "imdb_id": imdb_id,
                "title": title,
                "original_title": _clean_text(value("original_title")),
                "title_type": _clean_text(value("title_type")),
                "year": _number(value("year"), integer=True),
                "user_rating": user_rating,
                "date_rated": _clean_text(value("date_rated")),
                "imdb_rating": _number(value("imdb_rating")),
                "num_votes": _number(value("num_votes"), integer=True),
                "runtime_minutes": _number(value("runtime_minutes"), integer=True),
                "release_date": _clean_text(value("release_date")),
                "genres_csv": _clean_text(value("genres_csv")),
                "directors_csv": _clean_text(value("directors_csv")),
            }
        )

    if not records:
        raise ValueError("Aucune évaluation exploitable n’a été trouvée dans ce fichier.")
    return (
        pd.DataFrame.from_records(records)
        .drop_duplicates(subset=["imdb_id"], keep="last")
        .reset_index(drop=True)
    )


UPSERT = """
INSERT INTO titles (
    imdb_id, title, original_title, title_type, year, user_rating,
    date_rated, imdb_rating, num_votes, runtime_minutes, release_date,
    genres_csv, directors_csv
) VALUES (
    :imdb_id, :title, :original_title, :title_type, :year, :user_rating,
    :date_rated, :imdb_rating, :num_votes, :runtime_minutes, :release_date,
    :genres_csv, :directors_csv
)
ON CONFLICT(imdb_id) DO UPDATE SET
    title=excluded.title,
    original_title=COALESCE(excluded.original_title, titles.original_title),
    title_type=COALESCE(excluded.title_type, titles.title_type),
    year=COALESCE(excluded.year, titles.year),
    user_rating=excluded.user_rating,
    date_rated=COALESCE(excluded.date_rated, titles.date_rated),
    imdb_rating=COALESCE(excluded.imdb_rating, titles.imdb_rating),
    num_votes=COALESCE(excluded.num_votes, titles.num_votes),
    runtime_minutes=COALESCE(excluded.runtime_minutes, titles.runtime_minutes),
    release_date=COALESCE(excluded.release_date, titles.release_date),
    genres_csv=COALESCE(excluded.genres_csv, titles.genres_csv),
    directors_csv=COALESCE(excluded.directors_csv, titles.directors_csv)
"""


def import_ratings(
    source: str | Path | bytes | BinaryIO | TextIO,
    database: str | Path | None = None,
) -> ImportResult:
    initialize(database)
    raw = _read_csv(source)
    normalized = normalize_ratings(raw)

    with transaction(database) as connection:
        imported_ids = set(normalized["imdb_id"])
        existing = {
            row["imdb_id"]
            for row in connection.execute(
                "SELECT imdb_id FROM titles"
            ).fetchall()
            if row["imdb_id"] in imported_ids
        }
        connection.executemany(
            UPSERT,
            normalized.where(pd.notna(normalized), None).to_dict(orient="records"),
        )

    imported = len(normalized) - len(existing)
    return ImportResult(
        total_rows=len(raw),
        imported_rows=imported,
        updated_rows=len(existing),
        skipped_rows=len(raw) - len(normalized),
        columns=tuple(str(column) for column in raw.columns),
    )


def database_counts(database: str | Path | None = None) -> dict[str, int]:
    initialize(database)
    with connect(database) as connection:
        total = connection.execute("SELECT COUNT(*) FROM titles").fetchone()[0]
        enriched = connection.execute(
            "SELECT COUNT(*) FROM titles WHERE metadata_status='done'"
        ).fetchone()[0]
        pending = connection.execute(
            "SELECT COUNT(*) FROM titles WHERE metadata_status IN ('pending', 'error')"
        ).fetchone()[0]
    return {"total": total, "enriched": enriched, "pending": pending}
