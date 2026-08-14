from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from cineprofile.tmdb import TmdbClient
from cineprofile.vivier_audit import (
    latest_vivier_audit_report,
    run_vivier_audit,
    vivier_audit_report_path,
)


WINDOW_OPTIONS = {
    "Diagnostic — 3 fenêtres": 3,
    "Complet — 5 fenêtres": 5,
}


def _percentage(value: object) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def _load_report(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _render_report(path: Path, payload: dict) -> None:
    summary = payload.get("summary") or {}
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Films 8+ mesurables",
        int(summary.get("measurable_liked_films") or 0),
    )
    for column, budget in zip(metric_columns[1:], (100, 300, 500), strict=True):
        column.metric(
            f"Rappel à {budget}",
            _percentage(summary.get(f"recall_at_{budget}")),
        )
    st.caption(
        "Parmi les films 8+ réellement sortis dans la période par défaut : "
        + " · ".join(
            f"@{budget} {_percentage(summary.get(f'eligible_recall_at_{budget}'))}"
            for budget in (100, 300, 500)
        )
    )

    ablation = summary.get("source_ablation") or []
    if ablation:
        st.markdown("**Ce que chaque source apporte réellement**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Source": row["source"],
                        "Candidats bruts": row.get("raw_candidates", 0),
                        "Films 8+ touchés": row.get(
                            "targets_found_by_source", 0
                        ),
                        "Perdus sans elle @100": row.get(
                            "lost_hits_at_100", 0
                        ),
                        "Perdus sans elle @300": row.get(
                            "lost_hits_at_300", 0
                        ),
                        "Perdus sans elle @500": row.get(
                            "lost_hits_at_500", 0
                        ),
                    }
                    for row in ablation
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    missing = [
        row
        for window in payload.get("windows") or []
        for row in window.get("missing_traces") or []
    ]
    if missing:
        labels = {
            "absent_from_all_sources": "Absent de toutes les sources",
            "outside_release_window": "Hors période de sortie",
            "insufficient_votes": "Pas assez de votes",
            "excluded_genre": "Genre exclu",
            "beyond_analysis_budget": "Présent après la 500e place",
        }
        reasons = Counter(str(row.get("state")) for row in missing)
        st.markdown("**Pourquoi certains films 8+ manquent**")
        st.caption(
            " · ".join(
                f"{labels.get(reason, reason)} : {count}"
                for reason, count in reasons.most_common()
            )
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Film": row.get("title"),
                        "Note": row.get("rating"),
                        "Raison": labels.get(
                            str(row.get("state")), row.get("state")
                        ),
                        "Rang": row.get("rank"),
                        "Sources": ", ".join(row.get("sources") or []),
                    }
                    for row in missing[:100]
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    st.download_button(
        "Télécharger le rapport du vivier",
        data=path.read_bytes(),
        file_name=path.name,
        mime="application/json",
        width="stretch",
    )
    st.caption(
        "Le rapport ne contient aucune clé API. Il utilise des copies "
        "temporaires de la base et n’active aucune nouvelle règle."
    )


def render_vivier_audit_panel(
    database: str | Path,
    *,
    token: str,
    language: str,
    region: str,
    rated_count: int,
    logger: logging.Logger,
) -> None:
    with st.expander(
        "Mesurer le vivier — films 8+ retrouvés",
        expanded=False,
    ):
        st.write(
            "L’audit reconstruit plusieurs versions passées de ton profil, "
            "interroge les sources TMDB actuelles puis mesure combien de films "
            "que tu as ensuite notés 8+ étaient présents dans les 100, 300 et "
            "500 premiers candidats. Il retire aussi chaque source à tour de "
            "rôle pour mesurer son apport."
        )
        st.caption(
            "Les notes futures restent cachées. Les métadonnées et la "
            "popularité TMDB sont celles disponibles aujourd’hui."
        )
        option = st.selectbox(
            "Profondeur chronologique",
            list(WINDOW_OPTIONS),
            key="vivier_audit_windows",
        )
        if st.button(
            "Lancer la mesure du vivier",
            key="run_vivier_audit",
            type="primary",
            width="stretch",
            disabled=rated_count < 120 or not token,
            help=(
                "Il faut au moins 120 films datés et une connexion TMDB."
                if rated_count < 120 or not token
                else "Plusieurs centaines de requêtes TMDB peuvent être nécessaires."
            ),
        ):
            progress = st.progress(0.0)
            progress_text = st.empty()

            def on_progress(current: int, total: int, message: str) -> None:
                progress.progress(min(1.0, current / max(1, total)))
                progress_text.caption(message)

            try:
                with TmdbClient(
                    token,
                    language=language,
                    region=region,
                ) as client:
                    payload = run_vivier_audit(
                        client,
                        database,
                        requested_windows=WINDOW_OPTIONS[option],
                        on_progress=on_progress,
                    )
                path = vivier_audit_report_path(database, payload)
            except Exception as exc:
                logger.exception("vivier_audit_failed")
                st.error(f"La mesure du vivier a échoué : {exc}")
            else:
                progress.progress(1.0)
                progress_text.caption("Audit du vivier terminé.")
                st.session_state["latest_vivier_audit_path"] = str(path)
                st.success("Mesure terminée. Le moteur n’a pas été modifié.")

        path_value = st.session_state.get("latest_vivier_audit_path")
        path = (
            Path(str(path_value))
            if path_value
            else latest_vivier_audit_report(database)
        )
        if path is None or not path.is_file():
            return
        payload = _load_report(path)
        if payload is None:
            st.warning("Le dernier rapport du vivier est illisible.")
            return
        _render_report(path, payload)
