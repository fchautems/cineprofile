from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from cineprofile.preferences import (
    load_radarr_requests,
    upsert_radarr_catalog_entries,
    update_radarr_states,
)
from cineprofile.radarr import RadarrClient


def radarr_states_stale(
    requests: dict[int, dict],
    *,
    max_age_seconds: int = 25,
    now: datetime | None = None,
) -> bool:
    if not requests:
        return False
    current = now or datetime.now(UTC)
    threshold = current - timedelta(seconds=max_age_seconds)
    for row in requests.values():
        raw_checked_at = row.get("status_checked_at")
        if not raw_checked_at:
            return True
        try:
            checked_at = datetime.fromisoformat(str(raw_checked_at))
        except ValueError:
            return True
        if checked_at.tzinfo is None:
            checked_at = checked_at.replace(tzinfo=UTC)
        if checked_at < threshold:
            return True
    return False


def should_synchronize_radarr_states(
    requests: dict[int, dict],
    *,
    force_sync: bool = False,
    entered_my_list: bool = False,
    skip_once: bool = False,
    now: datetime | None = None,
) -> bool:
    """Decide whether this fragment run should contact Radarr.

    Entering Ma liste is deliberately stronger than the short freshness window:
    the screen must show a new technical snapshot straight away. A local filter
    or feedback interaction remains exempt from network work for that one run.
    """
    if not requests:
        return False
    if force_sync or entered_my_list:
        return True
    return not skip_once and radarr_states_stale(requests, now=now)


def synchronize_radarr_states(
    database: str | Path,
    radarr_config: dict,
) -> dict[int, dict]:
    """Fetch two Radarr endpoints once and persist the resulting snapshot."""
    requests = load_radarr_requests(database)
    if not requests:
        return requests
    with RadarrClient(
        radarr_config["url"],
        radarr_config["api_key"],
        timeout=10.0,
    ) as client:
        states = client.movie_states(set(requests))
    update_radarr_states(states, database)
    return load_radarr_requests(database)


def synchronize_radarr_catalog(
    database: str | Path,
    radarr_config: dict,
    recommendations: list[dict],
) -> dict[int, dict]:
    """Discover recommendation films already managed anywhere in Radarr."""
    if not recommendations:
        return load_radarr_requests(database)
    recommendation_ids = {
        int(item["tmdb_id"]) for item in recommendations if item.get("tmdb_id")
    }
    with RadarrClient(
        radarr_config["url"],
        radarr_config["api_key"],
        timeout=10.0,
    ) as client:
        catalog_states = client.all_movie_states()
    matching_states = {
        tmdb_id: state
        for tmdb_id, state in catalog_states.items()
        if tmdb_id in recommendation_ids
    }
    upsert_radarr_catalog_entries(recommendations, matching_states, database)
    return load_radarr_requests(database)
