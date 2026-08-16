"""Official IMDb rating snapshot used by recommendation cards and filters."""

from __future__ import annotations

import csv
import gzip
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

import httpx


IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
SNAPSHOT_MAX_AGE = timedelta(hours=24)


def snapshot_path(database: str | Path) -> Path:
    return Path(database).parent / "cache" / "imdb" / "title.ratings.tsv.gz"


def _is_fresh(path: Path, now: datetime) -> bool:
    if not path.is_file():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return now - modified <= SNAPSHOT_MAX_AGE


def ensure_snapshot(
    database: str | Path,
    *,
    now: datetime | None = None,
) -> Path:
    """Download IMDb's daily non-commercial rating snapshot when required."""
    target = snapshot_path(database)
    current_time = now or datetime.now(UTC)
    if _is_fresh(target, current_time):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=20.0),
            headers={"User-Agent": "CineProfile/0.17 IMDb ratings"},
        ) as client:
            with client.stream("GET", IMDB_RATINGS_URL) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_bytes():
                        output.write(chunk)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        if target.is_file():
            return target
        raise
    return target


def ratings_for_ids(
    path: str | Path,
    imdb_ids: Iterable[str],
) -> dict[str, tuple[float, int]]:
    """Read only requested titles from the compressed daily snapshot."""
    wanted = {str(imdb_id) for imdb_id in imdb_ids if str(imdb_id).startswith("tt")}
    if not wanted:
        return {}
    found: dict[str, tuple[float, int]] = {}
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            imdb_id = str(row.get("tconst") or "")
            if imdb_id not in wanted:
                continue
            found[imdb_id] = (
                float(row["averageRating"]),
                int(row["numVotes"]),
            )
            if len(found) == len(wanted):
                break
    return found


def hydrate_imdb_ratings(
    recommendations: Iterable[dict],
    database: str | Path,
) -> tuple[list[dict], int]:
    """Attach genuine IMDb ratings and vote counts to recommendations."""
    rows = [dict(item) for item in recommendations]
    missing_ids = {
        str(item.get("imdb_id") or "")
        for item in rows
        if item.get("imdb_id") and item.get("imdb_rating") is None
    }
    if not missing_ids:
        return rows, 0
    ratings = ratings_for_ids(ensure_snapshot(database), missing_ids)
    updated = 0
    for item in rows:
        rating = ratings.get(str(item.get("imdb_id") or ""))
        if rating is None:
            continue
        item["imdb_rating"], item["imdb_vote_count"] = rating
        updated += 1
    return rows, updated
