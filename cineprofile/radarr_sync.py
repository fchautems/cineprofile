from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from cineprofile.preferences import (
    load_radarr_requests,
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
