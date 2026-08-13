from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from cineprofile.preferences import (
    load_feedback,
    load_radarr_attempts,
    load_radarr_requests,
    record_radarr_attempt,
    record_radarr_download,
    remove_feedback,
    remove_radarr_request,
    save_feedback,
)
from cineprofile.radarr import RadarrClient


FEEDBACK_LABELS = {
    "watchlist": "À voir",
    "already_seen": "Déjà vu",
    "not_interested": "Pas intéressé",
}
LABEL_FEEDBACK = {label: action for action, label in FEEDBACK_LABELS.items()}
FILTERS = ("Tous", "À voir", "Déjà vus", "Pas intéressé", "Downloaded")


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
    elif status_filter == "Downloaded":
        filtered = [row for row in filtered if row["downloaded"]]
    needle = search.strip().casefold()
    if needle:
        filtered = [
            row
            for row in filtered
            if needle in str(row.get("title") or "").casefold()
        ]
    return filtered


def _clear_recommendation_state() -> None:
    for key in (
        "recommendations",
        "recommendation_lists",
        "recommendation_diagnostics",
    ):
        st.session_state.pop(key, None)


def _send_to_radarr(
    item: dict,
    database: str | Path,
    radarr_config: dict,
) -> None:
    try:
        with RadarrClient(
            radarr_config["url"],
            radarr_config["api_key"],
        ) as client:
            result = client.add_movie(
                int(item["tmdb_id"]),
                root_folder_path=radarr_config["root_folder_path"],
                quality_profile_id=int(radarr_config["quality_profile_id"]),
            )
    except Exception as exc:
        record_radarr_attempt(
            item,
            "failed",
            database,
            error_message=str(exc),
        )
        st.error(f"Envoi à Radarr échoué : {exc}")
    else:
        record_radarr_download(
            item,
            result.movie_id,
            database,
            already_present=result.already_present,
        )
        st.rerun()


def render_my_movies_tab(
    database: str | Path,
    *,
    radarr_config: dict | None = None,
) -> None:
    st.subheader("Mes films")
    st.write(
        "Retrouve tous les films marqués depuis les suggestions, modifie leur "
        "statut et consulte l’historique local des envois à Radarr."
    )
    movies = load_my_movies(database)
    if not movies:
        st.info(
            "Aucun film marqué pour le moment. Utilise À voir, Pas intéressé, "
            "Déjà vu ou Download depuis les suggestions."
        )
        return

    metrics = st.columns(4)
    metrics[0].metric("Tous", len(movies))
    metrics[1].metric(
        "À voir",
        sum(row["feedback_action"] == "watchlist" for row in movies),
    )
    metrics[2].metric(
        "Déjà vus",
        sum(row["feedback_action"] == "already_seen" for row in movies),
    )
    metrics[3].metric(
        "Downloaded",
        sum(row["downloaded"] for row in movies),
    )

    filter_column, search_column = st.columns([1, 2])
    status_filter = filter_column.selectbox("Afficher", FILTERS)
    search = search_column.text_input(
        "Chercher un film",
        placeholder="Titre…",
        key="my_movies_search",
    )
    visible = filter_my_movies(movies, status_filter, search)
    st.caption(f"{len(visible)} film(s)")
    if not visible:
        st.info("Aucun film ne correspond à ce filtre.")
        return

    frame = pd.DataFrame(
        [
            {
                "Film": row["title"],
                "Année": row["year"],
                "Statut": row["feedback_label"],
                "Downloaded": "Oui" if row["downloaded"] else "Non",
                "Mis à jour": str(row.get("updated_at") or "")[:19],
            }
            for row in visible
        ]
    )
    st.dataframe(frame, hide_index=True, width="stretch")

    labels = {
        f"{row['title']} ({row['year']}) · TMDB {row['tmdb_id']}": row
        for row in visible
    }
    selected_label = st.selectbox(
        "Modifier un film",
        list(labels),
        key="my_movies_selected",
    )
    selected = labels[selected_label]
    with st.container(border=True):
        st.markdown(f"### {selected['title']} ({selected['year']})")
        if selected.get("overview"):
            st.write(selected["overview"])
        status_options = ["Aucun", "À voir", "Déjà vu", "Pas intéressé"]
        current_label = selected["feedback_label"]
        current_index = (
            status_options.index(current_label)
            if current_label in status_options
            else 0
        )
        chosen_status = st.selectbox(
            "Statut CineProfile",
            status_options,
            index=current_index,
            key=f"my_movies_status_{selected['tmdb_id']}",
        )
        if st.button(
            "Enregistrer le statut",
            key=f"save_my_movie_{selected['tmdb_id']}",
        ):
            if chosen_status == "Aucun":
                remove_feedback(int(selected["tmdb_id"]), database)
            else:
                save_feedback(selected, LABEL_FEEDBACK[chosen_status], database)
            _clear_recommendation_state()
            st.rerun()

        st.divider()
        if selected["downloaded"]:
            st.success("Downloaded")
            st.caption(
                "Envoyé à Radarr au moins une fois — présence du fichier non "
                "vérifiée. Annuler ce statut agit uniquement dans CineProfile."
            )
            if st.button(
                "Annuler le statut Downloaded",
                key=f"clear_downloaded_{selected['tmdb_id']}",
            ):
                remove_radarr_request(int(selected["tmdb_id"]), database)
                st.rerun()
        else:
            st.caption(
                "Download envoie le film à Radarr, active sa surveillance et "
                "lance sa recherche."
            )
            if st.button(
                "Download",
                key=f"download_my_movie_{selected['tmdb_id']}",
                disabled=radarr_config is None,
                help=(
                    "Connecte d’abord Radarr dans la barre latérale."
                    if radarr_config is None
                    else "Envoyer ce film à Radarr."
                ),
            ):
                _send_to_radarr(selected, database, radarr_config)

    attempts = load_radarr_attempts(database, tmdb_id=int(selected["tmdb_id"]))
    with st.expander(
        f"Historique Radarr de ce film ({len(attempts)})",
        expanded=False,
    ):
        if not attempts:
            st.caption("Aucune tentative enregistrée.")
        else:
            outcome_labels = {
                "accepted": "Accepté par Radarr",
                "already_present": "Déjà présent dans Radarr",
                "failed": "Échec",
            }
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Date": str(row["attempted_at"])[:19],
                            "Résultat": outcome_labels[row["outcome"]],
                            "Détail": row.get("error_message") or "",
                        }
                        for row in attempts
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
