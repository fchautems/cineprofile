from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import streamlit as st

from cineprofile.genre_catalog import TMDB_EXCLUDABLE_GENRES
from cineprofile.ui_catalog import render_catalog_tab
from cineprofile.ui_connections import render_connections
from cineprofile.ui_import import render_import_tab
from cineprofile.ui_preferences import render_preferences_tab
from cineprofile.ui_profile import render_profile_tab
from cineprofile.ui_vivier_audit import render_vivier_audit_panel


def render_settings_tab(
    database: str | Path,
    environment_path: str | Path,
    *,
    token: str,
    language: str,
    region: str,
    counts: dict[str, int],
    profile: dict | None,
    clear_catalog_cache: Callable[[], None],
    logger: logging.Logger,
) -> tuple[dict[str, int], dict | None]:
    """Group infrequent configuration and maintenance actions in one place."""
    st.subheader("Réglages", anchor=False)
    st.caption("Connexions, préférences et maintenance de CineProfile.")

    st.markdown("### Connexions")
    render_connections(environment_path)
    st.caption(
        "Bazarr suit les films importés par Radarr ; aucune connexion directe "
        "n’est nécessaire pour les suggestions."
    )

    st.markdown("### Préférences")
    st.caption(f"Langue : {language} · Région : {region}")
    preference_columns = st.columns(2)
    preference_columns[0].selectbox(
        "Période par défaut",
        [
            "3 dernières années",
            "1 dernière année",
            "5 dernières années",
            "10 dernières années",
            "20 dernières années",
            "Toutes les années",
            "Période personnalisée",
        ],
        key="suggestion_period",
        persist_state="session",
    )
    preference_columns[1].selectbox(
        "Profondeur de recherche",
        ["Rapide", "Normale", "Approfondie"],
        key="suggestion_depth",
        persist_state="session",
    )

    st.markdown("### Maintenance")
    with st.expander("Importer IMDb et actualiser les données", icon=":material/upload_file:"):
        counts, profile = render_import_tab(
            database,
            token=token,
            language=language,
            region=region,
            counts=counts,
            profile=profile,
            clear_catalog_cache=clear_catalog_cache,
            logger=logger,
        )

    maintenance_actions = st.container(horizontal=True, gap="xsmall")
    if maintenance_actions.button(
        "Vider les caches",
        icon=":material/delete_sweep:",
        help="Vider les caches d’interface et de catalogue.",
    ):
        clear_catalog_cache()
        st.cache_data.clear()
        st.toast("Caches vidés.", icon=":material/check_circle:")
    if maintenance_actions.button(
        "Actualiser les statuts",
        icon=":material/sync:",
        help="Forcer l’actualisation Radarr à la prochaine ouverture de Ma liste.",
    ):
        st.session_state["force_radarr_sync"] = True
        st.toast("Actualisation Radarr demandée.", icon=":material/check_circle:")

    diagnostics = st.session_state.get("recommendation_diagnostics")
    if diagnostics:
        diagnostic_path = Path(str(diagnostics.get("diagnostic_path") or ""))
        if diagnostic_path.is_file():
            diagnostic_data = diagnostic_path.read_bytes()
            diagnostic_name = str(
                diagnostics.get("diagnostic_file_name") or diagnostic_path.name
            )
        else:
            diagnostic_data = json.dumps(
                diagnostics, ensure_ascii=False, indent=2
            ).encode("utf-8")
            diagnostic_name = "cineprofile-diagnostic.json"
        st.download_button(
            "Tester le moteur et télécharger le diagnostic",
            data=diagnostic_data,
            file_name=diagnostic_name,
            mime="application/json",
            icon=":material/download:",
        )

    show_audit = st.toggle(
        "Audit du vivier",
        value=False,
        help="Afficher l’audit complet uniquement quand il est nécessaire.",
    )
    if show_audit:
        render_vivier_audit_panel(
            database,
            token=token,
            language=language,
            region=region,
            rated_count=int(counts["total"]),
            logger=logger,
        )

    st.markdown("### Avancé")
    show_advanced = st.toggle(
        "Afficher les options avancées",
        value=False,
    )
    if show_advanced:
        st.session_state.setdefault("search_reliability", "Forte")
        st.session_state.setdefault("search_include_upcoming", False)
        st.session_state.setdefault("search_semantic_enabled", True)
        st.session_state.setdefault("search_include_classics", True)
        st.session_state.setdefault("search_excluded_genres", ["Horreur"])
        st.selectbox(
            "Fiabilité des avis TMDB",
            ["Souple", "Équilibrée", "Forte"],
            key="search_reliability",
        )
        st.toggle(
            "Inclure les films à venir",
            key="search_include_upcoming",
        )
        st.toggle(
            "Analyse sémantique locale",
            key="search_semantic_enabled",
        )
        st.toggle(
            "Préparer Classiques à découvrir",
            key="search_include_classics",
        )
        st.multiselect(
            "Genres exclus de la recherche",
            list(TMDB_EXCLUDABLE_GENRES),
            key="search_excluded_genres",
        )
        analysis_budget = st.number_input(
            "Budget d’analyse personnalisé (0 = automatique)",
            min_value=0,
            max_value=700,
            step=25,
            key="search_analysis_budget",
        )
        st.session_state["search_analysis_limit"] = (
            int(analysis_budget) or None
        )
        render_preferences_tab(database, profile)

        technical_log = Path(database).parent / "logs" / "cineprofile.log"
        if technical_log.is_file():
            st.download_button(
                "Télécharger le journal technique",
                data=technical_log.read_bytes(),
                file_name="cineprofile.log",
                mime="text/plain",
                width="stretch",
            )
        render_profile_tab(
            database,
            counts,
            profile,
            logger=logger,
            advanced=True,
        )
        render_catalog_tab(database)

    return counts, profile
