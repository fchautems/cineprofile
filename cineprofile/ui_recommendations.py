from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from cineprofile import __version__
from cineprofile.compat import CineProfileVersionMismatch, unpack_recommendation_run
from cineprofile.diagnostics import diagnostic_with_ui_view
from cineprofile.genre_catalog import TMDB_EXCLUDABLE_GENRES
from cineprofile.preferences import load_feedback
from cineprofile.ranking import build_recommendation_lists
from cineprofile.recommendation_state import load_saved_recommendations
from cineprofile.recommender import recommend_movies
from cineprofile.result_filters import filter_recommendations, sort_recommendations
from cineprofile.tmdb import TmdbClient
from cineprofile.ui_recommendation_cards import render_recommendation_cards


def _restore_saved_selection(database: str | Path, profile: dict | None) -> None:
    """Restore the last selection after a Streamlit restart.

    Films explicitly rejected or marked as seen since the search are removed
    from the restored view. The saved selection itself remains an audit trail.
    """
    if not profile or st.session_state.get("recommendations") is not None:
        return
    saved = load_saved_recommendations(profile.get("profile_run_id"), database)
    if not saved:
        return
    excluded = {
        tmdb_id
        for tmdb_id, feedback in load_feedback(database).items()
        if feedback.get("action") in {"not_interested", "already_seen"}
    }
    recommendations = [
        item
        for item in saved["recommendations"]
        if int(item.get("tmdb_id") or -1) not in excluded
    ]
    st.session_state["recommendations"] = recommendations
    st.session_state["recommendation_lists"] = build_recommendation_lists(
        recommendations
    )
    st.session_state["recommendation_diagnostics"] = saved["diagnostics"]
    st.session_state["recommendation_settings"] = saved["settings"]
    st.session_state["recommendation_updated_at"] = saved["updated_at"]


def _render_recommendation_list(
    database: str | Path,
    *,
    view: str,
    recommendations: list[dict],
    all_recommendations: list[dict],
    diagnostics: dict | None,
    diagnostic_download_payload: dict | None,
    diagnostic_download_name: str | None,
    radarr_config: dict | None,
) -> None:
    is_safe = view == "safe"
    key_prefix = f"recommendation_{view}"
    if not recommendations:
        st.info(
            "Aucune découverte distincte n’est disponible dans ce vivier."
            if not is_safe
            else "Aucune valeur sûre n’est disponible dans ce vivier."
        )
        return

    if is_safe:
        st.caption(
            "La qualité publique reste dominante, avec un garde-fou personnel "
            "pour écarter du haut de la liste les films qui ne te donnent "
            "manifestement pas envie."
        )
    else:
        tier_counts = {
            tier: sum(item.get("match_tier") == tier for item in recommendations)
            for tier in (
                "Correspondance solide",
                "Piste plausible",
                "Solution de repli",
            )
        }
        if tier_counts["Correspondance solide"]:
            st.success(
                f"{tier_counts['Correspondance solide']} correspondance(s) "
                "solide(s) détectée(s)."
            )
        else:
            st.warning(
                "Aucune correspondance solide dans ce vivier. Les premiers "
                "résultats sont des pistes personnalisées."
            )
        st.caption(
            f"{tier_counts['Piste plausible']} piste(s) plausible(s) · "
            f"{tier_counts['Solution de repli']} solution(s) de repli. "
            "Les dix premières valeurs sûres sont volontairement absentes."
        )
        learning = (diagnostics or {}).get("adaptive_learning", {})
        st.caption(
            "Apprentissage actif : les nouvelles notes IMDb réactualisent le "
            "profil, « À voir » affine l’envie et « Pas intéressé » devient "
            "un contre-exemple. Ces retours s’appliquent à la prochaine "
            "recherche."
            + (
                " "
                f"Signaux actuels : {int(learning.get('watchlist_signals', 0))} "
                f"à voir · {int(learning.get('negative_signals', 0))} refus."
                if learning
                else ""
            )
        )

    st.markdown("#### Filtrer et trier")
    available_genres = sorted(
        {
            genre
            for item in recommendations
            for genre in item.get("genres", [])
        }
    )
    available_platforms = sorted(
        {
            platform
            for item in recommendations
            for platforms in item.get("providers_ch", {}).values()
            for platform in platforms
        }
    )
    available_languages = sorted(
        {
            item["original_language"]
            for item in recommendations
            if item.get("original_language")
        }
    )

    minimum_public_rating = 0.0
    minimum_score = 0
    minimum_interest = 0
    if is_safe:
        filter_columns = st.columns(4)
        minimum_public_rating = filter_columns[0].slider(
            "Note publique corrigée minimale",
            0.0,
            10.0,
            0.0,
            0.1,
            key=f"{key_prefix}_minimum_public",
        )
        genre_column = filter_columns[1]
        platform_column = filter_columns[2]
        availability_column = filter_columns[3]
    else:
        filter_columns = st.columns(5)
        learned_results = any(
            item.get("personal_model_used") for item in recommendations
        )
        minimum_score = filter_columns[0].slider(
            (
                "Chance d’un 8+ minimale"
                if learned_results
                else "Affinité minimale"
            ),
            0,
            100,
            0,
            key=f"{key_prefix}_minimum_affinity",
        )
        minimum_interest = filter_columns[1].slider(
            "Envie minimale",
            0,
            100,
            0,
            key=f"{key_prefix}_minimum_interest",
            help="Indice d’attrait avant visionnage, distinct de la note.",
        )
        genre_column = filter_columns[2]
        platform_column = filter_columns[3]
        availability_column = filter_columns[4]

    wanted_genres = genre_column.multiselect(
        "Genres",
        available_genres,
        key=f"{key_prefix}_genres",
    )
    wanted_platforms = platform_column.multiselect(
        "Plateformes CH",
        available_platforms,
        key=f"{key_prefix}_platforms",
    )
    availability = availability_column.selectbox(
        "Disponibilité",
        [
            "Toutes",
            "Incluse/Gratuite",
            "Location/Achat",
            "Disponible en CH",
        ],
        key=f"{key_prefix}_availability",
    )

    secondary_filters = st.columns(3)
    wanted_languages = secondary_filters[0].multiselect(
        "Langues originales",
        available_languages,
        key=f"{key_prefix}_languages",
    )
    runtime_range = secondary_filters[1].slider(
        "Durée souhaitée",
        30,
        300,
        (30, 300),
        key=f"{key_prefix}_runtime_range",
        format="%d min",
        help=(
            "Les films sans durée connue restent visibles uniquement lorsque "
            "la plage complète 30–300 minutes est conservée."
        ),
    )
    learned_results = any(
        item.get("personal_model_used") for item in recommendations
    )
    if is_safe:
        sort_options = [
            "Ordre conseillé",
            "Note publique corrigée",
            "Fiabilité de la note publique",
            "Nombre de votes",
            "Date de sortie",
            "Durée",
            "Titre",
        ]
    else:
        sort_options = [
            "Ordre conseillé",
            "Indice d’envie",
            (
                "Chance d’un 8+"
                if learned_results
                else "Affinité personnelle"
            ),
            "Note personnelle prévue",
            "Confiance",
            "Note publique corrigée",
            "Nombre de votes",
            "Date de sortie",
            "Durée",
            "Titre",
        ]
    sort_choice = secondary_filters[2].selectbox(
        "Trier par",
        sort_options,
        index=0,
        key=f"{key_prefix}_sort",
    )
    descending = st.toggle(
        "Ordre décroissant",
        value=True,
        key=f"{key_prefix}_descending",
    )

    visible = filter_recommendations(
        recommendations,
        minimum_score=minimum_score,
        minimum_interest=minimum_interest,
        minimum_public_rating=minimum_public_rating,
        genres=set(wanted_genres),
        platforms=set(wanted_platforms),
        languages=set(wanted_languages),
        runtime_range=runtime_range,
        availability=availability,
    )

    sort_fields = {
        "Affinité personnelle": "affinity_index",
        "Chance d’un 8+": "like_probability",
        "Ordre conseillé": "recommended_rank",
        "Indice d’envie": "interest_score",
        "Note personnelle prévue": "predicted_rating",
        "Confiance": "confidence",
        "Note publique corrigée": "bayesian_rating",
        "Fiabilité de la note publique": "public_rating_reliability",
        "Nombre de votes": "vote_count",
        "Date de sortie": "release_date",
        "Durée": "runtime_minutes",
        "Titre": "title",
    }
    sort_field = sort_fields[sort_choice]
    visible = sort_recommendations(
        visible,
        field=sort_field,
        descending=False if sort_field == "recommended_rank" else descending,
    )

    filter_signature = hashlib.sha256(
        json.dumps(
            {
                "view": view,
                "score": minimum_score,
                "interest": minimum_interest,
                "public_rating": minimum_public_rating,
                "genres": wanted_genres,
                "platforms": wanted_platforms,
                "availability": availability,
                "languages": wanted_languages,
                "runtime": runtime_range,
                "sort": sort_choice,
                "descending": descending,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    signature_key = f"result_filter_signature_{view}"
    visible_count_key = f"visible_results_count_{view}"
    if st.session_state.get(signature_key) != filter_signature:
        st.session_state[signature_key] = filter_signature
        st.session_state[visible_count_key] = 20
    visible_count = int(st.session_state.get(visible_count_key, 20))
    st.caption(
        f"{len(visible)} suggestion(s) après filtrage · "
        f"{min(visible_count, len(visible))} affichée(s)"
    )

    if (
        diagnostics
        and diagnostic_download_payload is not None
        and diagnostic_download_name is not None
    ):
        ui_view = {
            "recommendation_view": view,
            "minimum_score": minimum_score,
            "minimum_interest": minimum_interest,
            "minimum_public_rating": minimum_public_rating,
            "genres": wanted_genres,
            "platforms": wanted_platforms,
            "availability": availability,
            "languages": wanted_languages,
            "runtime_range": list(runtime_range),
            "sort": sort_choice,
            "descending": descending,
            "visible_tmdb_ids": [
                int(item["tmdb_id"]) for item in visible
            ],
            "displayed_tmdb_ids": [
                int(item["tmdb_id"]) for item in visible[:visible_count]
            ],
        }
        view_payload = diagnostic_with_ui_view(
            diagnostic_download_payload,
            ui_view,
            view_recommendations=recommendations,
        )
        warning_checks = [
            row
            for row in view_payload.get("automated_checks", [])
            if row.get("status") == "warning"
        ]
        if warning_checks:
            st.warning(
                f"{len(warning_checks)} contrôle(s) automatique(s) "
                "concernent les résultats actuellement affichés."
            )
        st.download_button(
            "Tester le moteur et télécharger le diagnostic",
            data=json.dumps(
                view_payload,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
            file_name=diagnostic_download_name,
            mime="application/json",
            key=f"download_diagnostic_{view}_{diagnostics.get('search_id')}",
            width="stretch",
        )

    render_recommendation_cards(
        database,
        all_recommendations,
        visible,
        visible_count,
        view=view,
        radarr_config=radarr_config,
    )

    if visible_count < len(visible):
        if st.button(
            "Afficher 20 résultats supplémentaires",
            key=f"{key_prefix}_show_more",
            width="stretch",
        ):
            st.session_state[visible_count_key] = visible_count + 20
            st.rerun()


def render_recommendations_tab(
    database: str | Path,
    profile: dict | None,
    *,
    token: str,
    language: str,
    region: str,
    logger: logging.Logger,
    radarr_config: dict | None = None,
) -> None:
    st.subheader("Suggestions")
    st.write(
        "Une seule recherche produit maintenant deux listes complémentaires : "
        "les valeurs sûres privilégient les films publics solides qui restent "
        "compatibles avec toi, tandis que les découvertes explorent plus "
        "librement ton profil et la proximité des histoires."
    )
    with st.expander("Comprendre les deux listes", expanded=False):
        st.markdown(
            """
            **Valeurs sûres** garde la note publique corrigée comme signal
            principal, puis vérifie que le film atteint un minimum d’envie et
            de compatibilité avec toi. Un film unanimement apprécié ne doit
            donc plus arriver en tête s’il est manifestement hors de tes goûts.

            **Découvertes pour toi** part du même vivier, mais utilise ton
            historique, la proximité des histoires et l’indice d’envie. Les
            dix premières valeurs sûres en sont retirées afin que les deux
            listes jouent réellement des rôles différents.

            L’**indice d’envie** estime ton attrait avant visionnage. Il valorise
            les signatures et acteurs qui constituent une vraie accroche, les
            suites d’un épisode aimé et des traitements précis comme la comédie
            noire. Il peut aussi signaler des freins : biopic, histoire, sport,
            animation, équipe inconnue ou usure d’une franchise. Ce n’est pas
            une probabilité et ses réglages sont visibles dans **Préférences**.

            La **chance d’un 8+** estime séparément la probabilité que tu
            attribues au moins **8/10** après visionnage. Elle est calculée
            principalement à partir des films aux histoires les plus proches
            dans ton historique, en donnant autant de valeur aux contre-exemples
            qu’aux films aimés. Ta fréquence personnelle de 8+ sert de référence :
            30 % peut donc être une bonne chance si ta moyenne habituelle est 20 %.

            Un modèle global complète ce voisinage avec les thèmes, genres,
            réalisateurs, scénaristes et acteurs. Il apprend directement tes
            notes autour de ta propre moyenne : la note publique ne constitue
            plus le point de départ de la prédiction.

            TMDB intervient seulement pour écarter les fiches trop fragiles et,
            dans les découvertes, comme petit garde-fou de qualité.
            """
        )
    _restore_saved_selection(database, profile)
    if not profile:
        st.info("Calcule d’abord le profil.")
        recommendations = []
    elif not token:
        st.info("Le jeton TMDB est nécessaire pour rechercher des films.")
        recommendations = []
    else:
        existing_selection = st.session_state.get("recommendations", [])
        updated_at = st.session_state.get("recommendation_updated_at")
        if existing_selection and updated_at:
            st.caption(
                "Dernière sélection conservée · "
                + str(updated_at).replace("T", " ")[:19]
            )
        setting_columns = st.columns(2)
        period_choice = setting_columns[0].selectbox(
            "Période",
            [
                "3 dernières années",
                "1 dernière année",
                "5 dernières années",
                "10 dernières années",
                "20 dernières années",
                "Toutes les années",
                "Période personnalisée",
            ],
        )
        depth = setting_columns[1].selectbox(
            "Profondeur de recherche",
            ["Rapide", "Normale", "Approfondie"],
            index=1,
            help=(
                "Combine plusieurs sources. Une recherche approfondie est plus "
                "longue la première fois puis bénéficie du cache local."
            ),
        )

        today = date.today()
        with st.expander("Réglages avancés", expanded=False):
            advanced_columns = st.columns(3)
            reliability = advanced_columns[0].selectbox(
                "Fiabilité des avis TMDB",
                ["Souple", "Équilibrée", "Forte"],
                index=2,
                help=(
                    "Le minimum de votes s’adapte automatiquement à l’âge du "
                    "film. Les films récents sont moins pénalisés."
                ),
            )
            include_upcoming = advanced_columns[1].checkbox(
                "Inclure les films à venir",
                value=False,
            )
            semantic_enabled = advanced_columns[2].checkbox(
                "Analyse sémantique locale",
                value=True,
                help=(
                    "Compare localement les résumés aux films notés. En cas "
                    "d’indisponibilité, CineProfile utilise un repli lexical."
                ),
            )
            excluded_genre_names = st.multiselect(
                "Genres exclus de la recherche",
                list(TMDB_EXCLUDABLE_GENRES),
                default=["Horreur"],
                help=(
                    "Ces genres sont retirés avant même l’analyse détaillée. "
                    "L’horreur est exclue par défaut ; tu peux ajouter ou "
                    "retirer librement les autres genres."
                ),
            )
            excluded_genre_ids = {
                TMDB_EXCLUDABLE_GENRES[name]
                for name in excluded_genre_names
            }
            custom_budget = st.checkbox("Personnaliser le budget d’analyse")
            analysis_limit = (
                st.slider("Nombre maximal de fiches complètes", 50, 700, 300, 25)
                if custom_budget
                else None
            )

        custom_years: tuple[int, int] | None = None
        if period_choice == "Période personnalisée":
            maximum_custom_year = (
                today.year + 2 if include_upcoming else today.year
            )
            custom_years = st.slider(
                "Années de sortie",
                1900,
                maximum_custom_year,
                (2000, today.year),
            )

        period_years = {
            "1 dernière année": 1,
            "3 dernières années": 3,
            "5 dernières années": 5,
            "10 dernières années": 10,
            "20 dernières années": 20,
        }
        if custom_years:
            start_date = date(custom_years[0], 1, 1).isoformat()
            end_date = date(custom_years[1], 12, 31).isoformat()
        elif period_choice == "Toutes les années":
            start_date = None
            end_date = (
                (today + timedelta(days=730)).isoformat()
                if include_upcoming
                else today.isoformat()
            )
        else:
            start_date = (
                today - timedelta(days=365 * period_years[period_choice])
            ).isoformat()
            end_date = (
                (today + timedelta(days=730)).isoformat()
                if include_upcoming
                else today.isoformat()
            )

        search_label = (
            "Actualiser les suggestions"
            if existing_selection
            else "Créer mes suggestions"
        )
        if st.button(search_label, width="stretch"):
            try:
                with st.spinner(
                    "Construction du vivier, analyse des crédits et des histoires…"
                ):
                    with TmdbClient(
                        token,
                        language=language,
                        region=region,
                    ) as client:
                        recommendation_run = recommend_movies(
                            client,
                            profile,
                            database,
                            start_date=start_date,
                            end_date=end_date,
                            depth=depth,
                            reliability=reliability,
                            ranking_mode="Valeurs sûres",
                            exploration=None,
                            semantic_enabled=semantic_enabled,
                            analysis_limit=analysis_limit,
                            excluded_genre_ids=excluded_genre_ids,
                        )
                    recommendations, diagnostics = unpack_recommendation_run(
                        recommendation_run
                    )
            except CineProfileVersionMismatch as exc:
                st.error(str(exc))
            except Exception as exc:
                logger.exception(
                    "search_failed | app_version=%s",
                    __version__,
                )
                st.error(
                    "La recherche a été interrompue. Les résultats précédents "
                    f"sont conservés. Détail : {exc}"
                )
            else:
                st.session_state["recommendations"] = recommendations
                st.session_state["recommendation_lists"] = (
                    build_recommendation_lists(recommendations)
                )
                st.session_state["recommendation_diagnostics"] = diagnostics
                st.session_state["recommendation_settings"] = diagnostics.get(
                    "settings", {}
                )
                st.session_state["recommendation_updated_at"] = datetime.now(
                    UTC
                ).isoformat()
                st.session_state["visible_results_count_safe"] = 20
                st.session_state["visible_results_count_discovery"] = 20

        recommendations = st.session_state.get("recommendations", [])
        recommendation_lists = st.session_state.get("recommendation_lists")
        if recommendations and not recommendation_lists:
            recommendation_lists = build_recommendation_lists(recommendations)
            st.session_state["recommendation_lists"] = recommendation_lists
        diagnostics = st.session_state.get("recommendation_diagnostics")
        diagnostic_download_payload = None
        diagnostic_download_name = None
        if diagnostics:
            search_settings = diagnostics.get("settings", {})
            with st.expander("Diagnostic de cette recherche", expanded=False):
                st.write(
                    f"Période réellement appliquée : "
                    f"**{diagnostics['window_start']} → "
                    f"{diagnostics['window_end']}**"
                )
                diagnostic_columns = st.columns(4)
                diagnostic_columns[0].metric(
                    "Vivier unique",
                    diagnostics["unique_candidates"],
                )
                diagnostic_columns[1].metric(
                    "Déjà vus/refusés exclus",
                    int(diagnostics["excluded_already_seen_tmdb"])
                    + int(diagnostics["excluded_already_seen_imdb"])
                    + int(diagnostics.get("excluded_by_feedback", 0)),
                )
                diagnostic_columns[2].metric(
                    "Fiches en cache",
                    diagnostics["cache_hits"],
                )
                diagnostic_columns[3].metric(
                    "Films analysés",
                    diagnostics["returned"],
                )
                st.caption(
                    "Repérés par proximité d’histoire avant enrichissement : "
                    f"{int(diagnostics.get('semantic_retrieval_candidates', 0))} "
                    "· candidats issus uniquement de la popularité : "
                    f"{100 * float(diagnostics.get('popularity_only_selected_share', 0)):.0f}%."
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Étape avant enrichissement": "Hors période",
                                "Nombre": diagnostics.get(
                                    "excluded_outside_window", 0
                                ),
                            },
                            {
                                "Étape avant enrichissement": "Votes insuffisants",
                                "Nombre": diagnostics.get(
                                    "excluded_insufficient_votes", 0
                                ),
                            },
                            {
                                "Étape avant enrichissement": "Genres exclus",
                                "Nombre": diagnostics.get(
                                    "excluded_genres", 0
                                ),
                            },
                            {
                                "Étape avant enrichissement": "Candidats restants",
                                "Nombre": diagnostics.get(
                                    "after_pre_enrichment_filters", 0
                                ),
                            },
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Source": name,
                                "Candidats bruts": count,
                                "Retenus pour analyse": diagnostics.get(
                                    "selected_source_counts", {}
                                ).get(name, 0),
                            }
                            for name, count in diagnostics["source_counts"].items()
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
                personal_engine = diagnostics.get("personal_engine")
                if personal_engine:
                    st.caption(
                        "Moteur personnel utilisé pour tous les résultats : "
                        + (
                            str(diagnostics.get("personal_variant_label"))
                            if personal_engine == "personal_v09"
                            else {
                                "linear_v06": "v0.6 linéaire",
                                "islands_v07": "v0.7 par îlots",
                                "legacy_v05": "indice historique",
                            }.get(str(personal_engine), str(personal_engine))
                        )
                    )
                learning = diagnostics.get("adaptive_learning")
                if learning:
                    st.caption(
                        "Apprentissage progressif : "
                        f"{int(learning.get('watchlist_signals', 0))} film(s) "
                        "« À voir » comme signaux d’envie · "
                        f"{int(learning.get('negative_signals', 0))} refus "
                        "comme contre-exemples · "
                        f"{int(learning.get('seen_exclusions', 0))} film(s) "
                        "simplement exclus."
                    )
                engine = diagnostics.get("semantic_engine")
                st.caption(
                    "Proximité des histoires : "
                    + (
                        "modèle sémantique multilingue local"
                        if engine == "semantic"
                        else "repli lexical TF‑IDF"
                    )
                )
                if (
                    search_settings.get("semantic_enabled", semantic_enabled)
                    and engine != "semantic"
                ):
                    st.warning(
                        "Le modèle sémantique complet n’a pas pu être utilisé "
                        "pour cette recherche : le classement repose sur le "
                        "repli lexical, moins précis."
                    )
                if diagnostics.get("excluded_genre_ids"):
                    genre_names_by_id = {
                        genre_id: name
                        for name, genre_id in TMDB_EXCLUDABLE_GENRES.items()
                    }
                    st.caption(
                        "Genres exclus avant analyse : "
                        + ", ".join(
                            genre_names_by_id.get(int(genre_id), str(genre_id))
                            for genre_id in diagnostics["excluded_genre_ids"]
                        )
                    )
            diagnostic_path_value = diagnostics.get("diagnostic_path")
            if diagnostic_path_value:
                diagnostic_path = Path(str(diagnostic_path_value))
                if diagnostic_path.is_file():
                    diagnostic_payload = json.loads(
                        diagnostic_path.read_text(encoding="utf-8")
                    )
                    diagnostic_download_payload = diagnostic_payload
                    diagnostic_download_name = diagnostics.get(
                        "diagnostic_file_name",
                        diagnostic_path.name,
                    )

        if recommendation_lists:
            safe_tab, discovery_tab = st.tabs(
                ["✅ Valeurs sûres", "✨ Découvertes pour toi"]
            )
            with safe_tab:
                _render_recommendation_list(
                    database,
                    view="safe",
                    recommendations=recommendation_lists.get("safe", []),
                    all_recommendations=recommendations,
                    diagnostics=diagnostics,
                    diagnostic_download_payload=diagnostic_download_payload,
                    diagnostic_download_name=diagnostic_download_name,
                    radarr_config=radarr_config,
                )
            with discovery_tab:
                _render_recommendation_list(
                    database,
                    view="discovery",
                    recommendations=recommendation_lists.get(
                        "discovery", []
                    ),
                    all_recommendations=recommendations,
                    diagnostics=diagnostics,
                    diagnostic_download_payload=diagnostic_download_payload,
                    diagnostic_download_name=diagnostic_download_name,
                    radarr_config=radarr_config,
                )
