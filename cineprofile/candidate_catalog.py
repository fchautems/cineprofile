from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Callable

from .db import connect, initialize, transaction


CATALOG_CACHE_VERSION = 1
RECENT_CACHE_DAYS = 45
BACK_CATALOG_CACHE_DAYS = 180
CATALOG_START_YEAR = 1920


@dataclass(frozen=True)
class CachedCandidates:
    items: list[dict]
    cache_hit: bool


def date_segments(
    start_date: str | None,
    end_date: str | None,
) -> list[tuple[str, str]]:
    """Split broad discovery windows so one popular year cannot hide others."""
    end = date.fromisoformat(end_date) if end_date else date.today()
    start = (
        date.fromisoformat(start_date)
        if start_date
        else date(CATALOG_START_YEAR, 1, 1)
    )
    if start > end:
        return []
    years = end.year - start.year + 1
    if years <= 6:
        step = 1
    elif years <= 15:
        step = 2
    elif years <= 35:
        step = 5
    else:
        step = 10
    segments: list[tuple[str, str]] = []
    year = start.year
    while year <= end.year:
        segment_start = max(start, date(year, 1, 1))
        segment_end = min(end, date(min(end.year, year + step - 1), 12, 31))
        segments.append((segment_start.isoformat(), segment_end.isoformat()))
        year += step
    return segments


def back_catalog_segments(start_date: str | None) -> list[tuple[str, str]]:
    if not start_date:
        return []
    boundary = date.fromisoformat(start_date) - timedelta(days=1)
    if boundary.year < CATALOG_START_YEAR:
        return []
    return date_segments(
        date(CATALOG_START_YEAR, 1, 1).isoformat(),
        boundary.isoformat(),
    )


def _scan_key(payload: dict) -> str:
    encoded = json.dumps(
        {"cache_version": CATALOG_CACHE_VERSION, **payload},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fresh(completed_at: str, ttl_days: int) -> bool:
    try:
        completed = datetime.fromisoformat(completed_at)
    except (TypeError, ValueError):
        return False
    if completed.tzinfo is None:
        completed = completed.replace(tzinfo=UTC)
    return completed >= datetime.now(UTC) - timedelta(days=max(1, ttl_days))


def _load_scan(
    database: str | Path | None,
    scan_key: str,
    *,
    ttl_days: int,
) -> list[dict] | None:
    with connect(database) as connection:
        scan = connection.execute(
            """
            SELECT candidate_ids_json, completed_at, language, region
            FROM candidate_catalog_scans
            WHERE scan_key=?
            """,
            (scan_key,),
        ).fetchone()
        if scan is None or not _fresh(scan["completed_at"], ttl_days):
            return None
        try:
            candidate_ids = [
                int(value) for value in json.loads(scan["candidate_ids_json"])
            ]
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not candidate_ids:
            return []
        placeholders = ",".join("?" for _ in candidate_ids)
        rows = connection.execute(
            f"""
            SELECT tmdb_id, payload_json
            FROM candidate_catalog
            WHERE tmdb_id IN ({placeholders})
              AND language=? AND region=?
            """,
            [*candidate_ids, scan["language"], scan["region"]],
        ).fetchall()
    payloads: dict[int, dict] = {}
    for row in rows:
        try:
            payloads[int(row["tmdb_id"])] = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError):
            return None
    if len(payloads) != len(candidate_ids):
        return None
    return [payloads[candidate_id] for candidate_id in candidate_ids]


def _save_scan(
    database: str | Path | None,
    *,
    scan_key: str,
    language: str,
    region: str,
    source: str,
    start_date: str | None,
    end_date: str | None,
    items: list[dict],
) -> None:
    now = datetime.now(UTC).isoformat()
    unique = {int(item["id"]): dict(item) for item in items}
    with transaction(database) as connection:
        connection.executemany(
            """
            INSERT INTO candidate_catalog(
              tmdb_id, language, region, release_date, payload_json,
              first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tmdb_id, language, region) DO UPDATE SET
              release_date=excluded.release_date,
              payload_json=excluded.payload_json,
              last_seen_at=excluded.last_seen_at
            """,
            [
                (
                    candidate_id,
                    language,
                    region,
                    item.get("release_date"),
                    json.dumps(item, ensure_ascii=False),
                    now,
                    now,
                )
                for candidate_id, item in unique.items()
            ],
        )
        connection.execute(
            """
            INSERT INTO candidate_catalog_scans(
              scan_key, language, region, source, start_date, end_date,
              candidate_ids_json, candidate_count, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scan_key) DO UPDATE SET
              candidate_ids_json=excluded.candidate_ids_json,
              candidate_count=excluded.candidate_count,
              completed_at=excluded.completed_at
            """,
            (
                scan_key,
                language,
                region,
                source,
                start_date,
                end_date,
                json.dumps(list(unique), separators=(",", ":")),
                len(unique),
                now,
            ),
        )


def _cached_call(
    client,
    database: str | Path | None,
    *,
    source: str,
    start_date: str | None,
    end_date: str | None,
    parameters: dict,
    loader: Callable[[], list[dict]],
    ttl_days: int,
) -> CachedCandidates:
    initialize(database)
    language = str(getattr(client, "language", "fr-FR"))
    region = str(getattr(client, "region", "CH"))
    key = _scan_key(
        {
            "language": language,
            "region": region,
            "source": source,
            "start_date": start_date,
            "end_date": end_date,
            "parameters": parameters,
        }
    )
    cached = _load_scan(database, key, ttl_days=ttl_days)
    if cached is not None:
        return CachedCandidates(cached, True)
    items = loader()
    _save_scan(
        database,
        scan_key=key,
        language=language,
        region=region,
        source=source,
        start_date=start_date,
        end_date=end_date,
        items=items,
    )
    return CachedCandidates(items, False)


def cached_discover(
    client,
    database: str | Path | None,
    *,
    source: str,
    start_date: str | None,
    end_date: str | None,
    pages: int,
    min_votes: int,
    sort_by: str = "popularity.desc",
    ttl_days: int = RECENT_CACHE_DAYS,
    **filters: int | None,
) -> CachedCandidates:
    parameters = {
        "pages": int(pages),
        "min_votes": int(min_votes),
        "sort_by": sort_by,
        **{key: value for key, value in filters.items() if value is not None},
    }
    return _cached_call(
        client,
        database,
        source=source,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters,
        loader=lambda: client.discover_recent_movies(
            start_date,
            end_date,
            pages=pages,
            min_votes=min_votes,
            sort_by=sort_by,
            regional_release_dates=False,
            **filters,
        ),
        ttl_days=ttl_days,
    )


def cached_seed_movies(
    client,
    database: str | Path | None,
    *,
    source: str,
    tmdb_id: int,
    pages: int,
    method: str = "movie_recommendations",
    ttl_days: int = RECENT_CACHE_DAYS,
) -> CachedCandidates:
    return _cached_call(
        client,
        database,
        source=source,
        start_date=None,
        end_date=None,
        parameters={
            "method": method,
            "tmdb_id": int(tmdb_id),
            "pages": int(pages),
        },
        loader=lambda: getattr(client, method)(int(tmdb_id), pages=pages),
        ttl_days=ttl_days,
    )
