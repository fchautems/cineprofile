from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable

import streamlit as st

from cineprofile.imdb_import import database_counts, import_ratings
from cineprofile.profile import build_profile
from cineprofile.tmdb import (
    TmdbClient,
    enrich_library,
    refresh_library_metadata,
)


def render_import_tab(
    database: str | Path,
    *,
    token: str,
    language: str,
    region: str,
    counts: dict[str, int],
    profile: dict | None,
    clear_catalog_cache: Callable[[], None],
    logger: logging.Logger,
) -> tuple[dict[str, int], dict | None]:
    st.subheader("Importer l’historique IMDb")
    uploaded = st.file_uploader(
        "Dépose le fichier ratings.csv",
        type=["csv"],
        help="Un nouvel import met à jour les notes sans effacer l’enrichissement existant.",
    )
    if uploaded is not None:
        payload = uploaded.getvalue()
        upload_digest = hashlib.sha256(payload).hexdigest()
        if st.session_state.get("import_digest") != upload_digest:
            st.session_state.pop("import_result", None)
            try:
                result = import_ratings(payload, database)
                counts = database_counts(database)
                profile = build_profile(database)
            except Exception as exc:
                logger.exception("import_failed")
                st.error(f"L’import a échoué : {exc}")
            else:
                st.session_state["import_digest"] = upload_digest
                st.session_state["import_result"] = result
                clear_catalog_cache()
        result = st.session_state.get("import_result")
        if result:
            st.success("Import IMDb terminé sans effacer les données existantes.")
            import_metrics = st.columns(4)
            import_metrics[0].metric("Nouveaux", result.imported_rows)
            import_metrics[1].metric("Modifiés", result.updated_rows)
            import_metrics[2].metric("Inchangés", result.unchanged_rows)
            import_metrics[3].metric("Ignorés", result.skipped_rows)
            st.caption(
                "Conservés pendant l’import : "
                f"{result.enriched_rows_preserved} enrichissement(s) TMDB · "
                f"{result.feedback_rows_preserved} statut(s) de film · "
                f"{result.preference_rows_preserved} correction(s) du profil."
            )
            st.caption("Colonnes détectées : " + ", ".join(result.columns))

    st.divider()
    st.subheader("Enrichissement")
    st.write(
        "TMDB complète chaque identifiant IMDb avec les crédits, thèmes, "
        "descriptions, langues, pays, sociétés, affiches et disponibilités. "
        "La base enregistre chaque résultat : une interruption ne fait pas "
        "recommencer les films déjà traités."
    )
    left, right = st.columns(2)
    batch_50 = left.button(
        "Tester sur 50 titres",
        disabled=not token or counts["pending"] == 0,
        width="stretch",
    )
    batch_all = right.button(
        "Enrichir tout l’historique",
        disabled=not token or counts["pending"] == 0,
        width="stretch",
    )
    if not token:
        st.info(
            "Ajoute un jeton TMDB dans la barre latérale pour enrichir les données. "
            "L’analyse des colonnes IMDb fonctionne déjà sans jeton."
        )

    if batch_50 or batch_all:
        progress = st.progress(0)
        message = st.empty()

        def on_progress(item) -> None:
            progress.progress(item.processed / max(item.total, 1))
            message.caption(
                f"{item.processed}/{item.total} · {item.title} · {item.status}"
            )

        try:
            with TmdbClient(token, language=language, region=region) as client:
                enrich_result = enrich_library(
                    client,
                    database,
                    limit=50 if batch_50 else None,
                    on_progress=on_progress,
                )
            counts = database_counts(database)
            profile = build_profile(database)
        except Exception as exc:
            logger.exception("enrichment_failed")
            st.error(f"L’enrichissement a été interrompu : {exc}")
        else:
            clear_catalog_cache()
            st.success(
                f"Terminé : {enrich_result['done']} enrichis, "
                f"{enrich_result['not_found']} absents de TMDB, "
                f"{enrich_result['error']} erreurs."
            )
            st.rerun()

    with st.expander("Maintenance des traductions", expanded=False):
        st.write(
            "Actualise les fiches déjà enrichies avec les textes `fr-FR`. "
            "Ce n’est pas nécessaire pour passer à la nouvelle version."
        )
        confirm_refresh = st.checkbox(
            "Je veux réellement actualiser toutes les fiches",
            key="confirm_metadata_refresh",
        )
        refresh_requested = st.button(
            "Actualiser les textes français",
            disabled=not token or not confirm_refresh or counts["enriched"] == 0,
        )
        if refresh_requested:
            refresh_progress = st.progress(0)
            refresh_message = st.empty()

            def on_refresh_progress(item) -> None:
                refresh_progress.progress(item.processed / max(item.total, 1))
                refresh_message.caption(
                    f"{item.processed}/{item.total} · {item.title} · {item.status}"
                )

            try:
                with TmdbClient(token, language=language, region=region) as client:
                    refresh_result = refresh_library_metadata(
                        client,
                        database,
                        on_progress=on_refresh_progress,
                    )
                counts = database_counts(database)
                profile = build_profile(database)
            except Exception as exc:
                logger.exception("metadata_refresh_failed")
                st.error(f"L’actualisation a été interrompue : {exc}")
            else:
                clear_catalog_cache()
                st.success(
                    f"{refresh_result['done']} fiches actualisées, "
                    f"{refresh_result['error']} erreurs."
                )

    return counts, profile
