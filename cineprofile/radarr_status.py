"""User-facing Radarr status semantics shared by Suggestions and Ma liste."""

from __future__ import annotations


ACTIVE_RADARR_STATES = frozenset(
    {"sent", "monitored", "no_result", "downloading", "available", "downloaded"}
)

RADARR_STATUS_META = {
    "sent": ("violet", ":material/radar:", "Surveillé"),
    "monitored": ("violet", ":material/radar:", "Surveillé"),
    "no_result": ("violet", ":material/radar:", "Surveillé"),
    "downloading": ("orange", ":material/downloading:", "Téléchargement en cours"),
    "available": ("blue", ":material/event_available:", "Disponible"),
    "downloaded": ("green", ":material/download_done:", "Téléchargé"),
    "error": ("red", ":material/error:", "Erreur"),
    "missing": ("gray", ":material/cloud_off:", "Pas encore envoyé"),
    "unmonitored": ("gray", ":material/notifications_off:", "Non surveillé"),
}


def radarr_status_meta(state: object) -> tuple[str, str, str]:
    return RADARR_STATUS_META.get(str(state or "missing"), RADARR_STATUS_META["missing"])


def active_radarr_ids(requests: dict[int, dict]) -> set[int]:
    return {
        int(tmdb_id)
        for tmdb_id, row in requests.items()
        if str(row.get("radarr_state") or "sent") in ACTIVE_RADARR_STATES
    }


def without_active_radarr_movies(
    recommendations: list[dict],
    requests: dict[int, dict],
) -> list[dict]:
    excluded = active_radarr_ids(requests)
    return [
        item
        for item in recommendations
        if int(item.get("tmdb_id") or -1) not in excluded
    ]
