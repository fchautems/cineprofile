from __future__ import annotations

from pathlib import Path

import streamlit as st

from cineprofile.preferences import (
    load_feedback,
    load_radarr_requests,
)
from cineprofile.radarr_sync import (
    should_synchronize_radarr_states,
    synchronize_radarr_states,
)
from cineprofile.ui_recommendation_cards import render_recommendation_cards


FEEDBACK_LABELS = {
    "watchlist": "À voir",
    "already_seen": "Déjà vu",
    "not_interested": "Pas intéressé",
}
FILTERS = ("Tous", "À voir", "Déjà vus", "Pas intéressé", "Radarr")
FILTER_ICONS = {
    "Tous": ":material/format_list_bulleted:",
    "À voir": ":material/thumb_up:",
    "Déjà vus": ":material/visibility:",
    "Pas intéressé": ":material/thumb_down:",
    "Radarr": ":material/radar:",
}
FILTER_HELP = {
    "Tous": "Tous les films de Ma liste",
    "À voir": "À voir",
    "Déjà vus": "Déjà vus",
    "Pas intéressé": "Pas pour moi",
    "Radarr": "Envoyés à Radarr",
}


def load_my_movies(database: str | Path) -> list[dict]:
    feedback = load_feedback(database)
    downloads = load_radarr_requests(database)
    movie_ids = set(feedback) | set(downloads)
    movies: list[dict] = []
    for tmdb_id in movie_ids:
        feedback_row = feedback.get(tmdb_id, {})
        download_row = downloads.get(tmdb_id, {})
        payload = dict(download_row.get("payload") or {})
        payload.update(feedback_row.get("payload") or {})
        title = str(
            feedback_row.get("title")
            or download_row.get("title")
            or payload.get("title")
            or f"TMDB {tmdb_id}"
        )
        release_date = str(payload.get("release_date") or "")
        year = release_date[:4] or payload.get("year") or "—"
        feedback_action = feedback_row.get("action")
        updated_values = [
            str(value)
            for value in (
                feedback_row.get("updated_at"),
                download_row.get("updated_at"),
            )
            if value
        ]
        movies.append(
            {
                **payload,
                "tmdb_id": int(tmdb_id),
                "imdb_id": feedback_row.get("imdb_id")
                or download_row.get("imdb_id")
                or payload.get("imdb_id"),
                "title": title,
                "year": year,
                "feedback_action": feedback_action,
                "feedback_label": FEEDBACK_LABELS.get(feedback_action, "—"),
                "downloaded": bool(download_row),
                "downloaded_at": download_row.get("requested_at"),
                "radarr_state": download_row.get("radarr_state"),
                "radarr_status_detail": download_row.get("status_detail"),
                "radarr_progress": download_row.get("progress"),
                "radarr_checked_at": download_row.get("status_checked_at"),
                "updated_at": max(updated_values) if updated_values else "",
            }
        )
    return sorted(
        movies,
        key=lambda row: (str(row.get("updated_at") or ""), row["title"]),
        reverse=True,
    )


def filter_my_movies(
    movies: list[dict],
    status_filter: str,
    search: str = "",
) -> list[dict]:
    if status_filter not in FILTERS:
        raise ValueError("Filtre de films non reconnu.")
    filtered = movies
    if status_filter == "À voir":
        filtered = [row for row in filtered if row["feedback_action"] == "watchlist"]
    elif status_filter == "Déjà vus":
        filtered = [
            row for row in filtered if row["feedback_action"] == "already_seen"
        ]
    elif status_filter == "Pas intéressé":
        filtered = [
            row for row in filtered if row["feedback_action"] == "not_interested"
        ]
    elif status_filter == "Radarr":
        filtered = [row for row in filtered if row["downloaded"]]
    needle = search.strip().casefold()
    if needle:
        filtered = [
            row
            for row in filtered
            if needle in str(row.get("title") or "").casefold()
        ]
    return filtered


def _select_filter(filter_name: str) -> None:
    st.session_state["my_movies_filter"] = filter_name
    st.session_state["skip_radarr_sync_once"] = True


def _skip_radarr_sync_once() -> None:
    st.session_state["skip_radarr_sync_once"] = True


def _request_radarr_sync() -> None:
    st.session_state["force_radarr_sync"] = True


@st.fragment(run_every="30s")
def render_my_movies_tab(
    database: str | Path,
    *,
    radarr_config: dict | None = None,
    refresh_on_open: bool = False,
) -> None:
    radarr_requests = load_radarr_requests(database)
    force_sync = bool(st.session_state.pop("force_radarr_sync", False))
    skip_sync = bool(st.session_state.pop("skip_radarr_sync_once", False))
    needs_sync = radarr_config and should_synchronize_radarr_states(
        radarr_requests,
        force_sync=force_sync,
        entered_my_list=refresh_on_open,
        skip_once=skip_sync,
    )
    if needs_sync:
        with st.spinner("Actualisation des états Radarr…"):
            try:
                synchronize_radarr_states(database, radarr_config)
            except Exception as exc:
                st.session_state["radarr_sync_error"] = str(exc)
            else:
                st.session_state.pop("radarr_sync_error", None)

    st.subheader("Ma liste")
    st.write(
        "Retrouve tous les films sur lesquels tu as agi dans CineProfile, "
        "y compris ceux envoyés à Radarr."
    )
    movies = load_my_movies(database)
    if not movies:
        st.info(
            "Aucun film marqué pour le moment. Utilise À voir, Pas intéressé, "
            "Déjà vu ou Envoyer à Radarr depuis les suggestions."
        )
        return

    if radarr_requests:
        status_column, refresh_column = st.columns([8, 1])
        if st.session_state.get("radarr_sync_error"):
            status_column.warning(
                "Radarr est momentanément injoignable. Les derniers états "
                "connus restent affichés."
            )
        else:
            checked = [
                str(row.get("status_checked_at") or "")
                for row in load_radarr_requests(database).values()
                if row.get("status_checked_at")
            ]
            if checked:
                status_column.caption(
                    "États Radarr actualisés à " + max(checked)[11:19]
                )
        refresh_column.button(
            ":material/sync:",
            key="refresh_radarr_states",
            help="Actualiser maintenant les états Radarr",
            width="stretch",
            on_click=_request_radarr_sync,
        )

    counts = {
        "Tous": len(movies),
        "À voir": sum(row["feedback_action"] == "watchlist" for row in movies),
        "Déjà vus": sum(
            row["feedback_action"] == "already_seen" for row in movies
        ),
        "Pas intéressé": sum(
            row["feedback_action"] == "not_interested" for row in movies
        ),
        "Radarr": sum(row["downloaded"] for row in movies),
    }
    status_filter = st.session_state.get("my_movies_filter", "Tous")
    filter_columns = st.columns(len(FILTERS))
    for column, filter_name in zip(filter_columns, FILTERS, strict=True):
        column.button(
            f"{FILTER_ICONS[filter_name]} {counts[filter_name]}",
            key=f"my_movies_filter_{filter_name}",
            type="primary" if status_filter == filter_name else "secondary",
            width="stretch",
            help=FILTER_HELP[filter_name],
            on_click=_select_filter,
            args=(filter_name,),
        )

    search = st.text_input(
        "Chercher un film",
        placeholder="Titre…",
        key="my_movies_search",
        on_change=_skip_radarr_sync_once,
    )
    visible = filter_my_movies(movies, status_filter, search)
    st.caption(f"{len(visible)} film(s)")
    if not visible:
        st.info("Aucun film ne correspond à ce filtre.")
        return

    render_recommendation_cards(
        database,
        movies,
        visible,
        len(visible),
        view="my_list",
        radarr_config=radarr_config,
    )
