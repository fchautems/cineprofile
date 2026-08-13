from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from cineprofile.preferences import (
    load_feedback,
    load_radarr_requests,
    record_radarr_attempt,
    record_radarr_download,
    save_feedback,
)
from cineprofile.radarr import RadarrClient


def _remove_from_recommendation_state(tmdb_id: int) -> None:
    st.session_state["recommendations"] = [
        row
        for row in st.session_state.get("recommendations", [])
        if int(row["tmdb_id"]) != tmdb_id
    ]
    recommendation_lists = st.session_state.get("recommendation_lists")
    if isinstance(recommendation_lists, dict):
        st.session_state["recommendation_lists"] = {
            name: [
                row
                for row in rows
                if int(row["tmdb_id"]) != tmdb_id
            ]
            for name, rows in recommendation_lists.items()
        }


def render_recommendation_cards(
    database: str | Path,
    recommendations: list[dict],
    visible: list[dict],
    visible_count: int,
    *,
    view: str = "discovery",
    radarr_config: dict | None = None,
) -> None:
    is_safe = view == "safe"
    feedback = load_feedback(database)
    radarr_requests = load_radarr_requests(database)
    for item in visible[:visible_count]:
        with st.container(border=True):
            poster_col, title_col, score_col = st.columns([1.05, 4.5, 1.25])
            imdb_url = (
                f"https://www.imdb.com/title/{item['imdb_id']}/"
                if item.get("imdb_id")
                else f"https://www.themoviedb.org/movie/{item['tmdb_id']}"
            )
            external_site = "IMDb" if item.get("imdb_id") else "TMDB"
            if item.get("poster_path"):
                poster_url = (
                    "https://image.tmdb.org/t/p/w342"
                    + item["poster_path"]
                )
                poster_col.markdown(
                    f'<a href="{html.escape(imdb_url)}" target="_blank">'
                    f'<img src="{html.escape(poster_url)}" '
                    'style="width:100%;border-radius:12px" '
                    f'alt="Ouvrir la fiche {external_site}"></a>',
                    unsafe_allow_html=True,
                )
                poster_col.caption(f"Cliquer pour ouvrir {external_site}")
            else:
                poster_col.caption("Affiche indisponible")

            title_col.markdown(
                f"### {item['title']} "
                f"({(item.get('release_date') or '—')[:4]})"
            )
            if item.get("match_tier"):
                title_col.caption("Niveau : " + str(item["match_tier"]))
            title_col.write(item.get("overview") or "Résumé non disponible.")
            if item.get("directors"):
                title_col.markdown(
                    "**Réalisation :** " + ", ".join(item["directors"])
                )
            if item.get("cast"):
                title_col.markdown(
                    "**Acteurs principaux :** " + ", ".join(item["cast"])
                )
            metadata_parts = [
                " · ".join(item.get("genres", [])),
                (
                    f"{item['runtime_minutes']} min"
                    if item.get("runtime_minutes")
                    else ""
                ),
                (
                    f"TMDB {float(item['vote_average']):.1f}/10 "
                    f"sur {int(item.get('vote_count') or 0):,} votes"
                    .replace(",", "’")
                    if item.get("vote_average")
                    else ""
                ),
            ]
            title_col.caption(
                " · ".join(part for part in metadata_parts if part)
            )
            if item.get("sources"):
                title_col.caption(
                    "Trouvé via : " + " · ".join(item["sources"])
                )
            if not is_safe:
                positive_interest = item.get("interest_positive_reasons", [])
                interest_reservations = item.get("interest_reservations", [])
                if positive_interest:
                    title_col.markdown(
                        "**Pourquoi pour toi :** "
                        + " · ".join(
                            str(row["label"]) for row in positive_interest[:3]
                        )
                    )
                if interest_reservations:
                    title_col.markdown(
                        "**Freins possibles :** "
                        + " · ".join(
                            str(row["label"])
                            for row in interest_reservations[:3]
                        )
                    )
                for reason in item.get("reasons", []):
                    if reason.startswith(
                        (
                            "Pourquoi il peut donner envie :",
                            "Freins possibles :",
                        )
                    ):
                        continue
                    title_col.caption("• " + reason)
            title_col.link_button(
                f"Voir sur {external_site}",
                imdb_url,
                key=f"external_{view}_{item['tmdb_id']}",
            )

            if is_safe:
                public_rating = float(
                    item.get(
                        "bayesian_rating",
                        item.get("public_rating_adjusted") or 0.0,
                    )
                )
                score_col.metric(
                    "Note publique corrigée",
                    f"{public_rating:.1f}/10",
                )
                score_col.caption(
                    "Fiabilité "
                    f"{float(item.get('public_rating_reliability') or 0):.0f}%"
                )
                score_col.caption(
                    f"{int(item.get('vote_count') or 0):,} votes".replace(
                        ",", "’"
                    )
                )
                if item.get("safe_eligibility_label"):
                    score_col.caption(
                        str(item["safe_eligibility_label"])
                    )
            elif item.get("interest_score") is not None:
                score_col.metric(
                    "Envie probable",
                    str(item.get("interest_label") or "—"),
                )
                score_col.caption(
                    f"Indice d’envie {float(item['interest_score']):.0f}/100"
                )
            elif item.get("personal_model_used"):
                score_col.metric(
                    "Chance d’un 8+",
                    f"{item['like_probability']:.0f} %",
                )
            else:
                score_col.metric(
                    "Affinité personnelle",
                    f"{item['affinity_index']:.0f}/100",
                )
            if item.get("personal_model_used") and not is_safe:
                base_rate = float(
                    item.get("base_like_rate_percent")
                    or 100.0 * float(item.get("base_like_rate") or 0.0)
                )
                probability = float(item["like_probability"])
                lift_points = float(
                    item.get("like_probability_lift_points")
                    or probability - base_rate
                )
                score_col.caption(
                    f"Chance d’un 8+ : {probability:.0f}% "
                    f"(habitude : {base_rate:.0f}%)"
                )
                score_col.caption(
                    f"Gain estimé : {lift_points:+.0f} points"
                )
                score_col.caption(
                    f"Note prévue {item['predicted_rating']:.1f}/10 "
                    f"· fourchette {item['prediction_low']:.1f}–"
                    f"{item['prediction_high']:.1f}"
                )
            score_col.caption(
                f"Rang « {item.get('ranking_mode', 'Valeurs sûres')} » : "
                f"{int(item.get('recommended_rank') or 0)}"
            )
            if not is_safe:
                score_col.caption(
                    f"Confiance {item['confidence_label']} "
                    f"({item['confidence']:.0f}/100)"
                )
            existing_feedback = feedback.get(int(item["tmdb_id"]))
            if existing_feedback:
                score_col.info(
                    {
                        "watchlist": "À voir",
                        "not_interested": "Pas intéressé",
                        "already_seen": "Déjà vu",
                    }[existing_feedback["action"]]
                )

            radarr_request = radarr_requests.get(int(item["tmdb_id"]))
            if radarr_request:
                score_col.success("Downloaded")

            action_columns = st.columns(4)
            if action_columns[0].button(
                "À voir",
                key=f"watchlist_{view}_{item['tmdb_id']}",
                width="stretch",
            ):
                save_feedback(item, "watchlist", database)
                st.rerun()
            if action_columns[1].button(
                "Pas intéressé",
                key=f"reject_{view}_{item['tmdb_id']}",
                width="stretch",
            ):
                save_feedback(item, "not_interested", database)
                _remove_from_recommendation_state(int(item["tmdb_id"]))
                st.rerun()
            if action_columns[2].button(
                "Déjà vu",
                key=f"seen_{view}_{item['tmdb_id']}",
                width="stretch",
            ):
                save_feedback(item, "already_seen", database)
                _remove_from_recommendation_state(int(item["tmdb_id"]))
                st.rerun()
            if action_columns[3].button(
                "Downloaded" if radarr_request else "Download",
                key=f"radarr_{view}_{item['tmdb_id']}",
                width="stretch",
                disabled=radarr_config is None or radarr_request is not None,
                help=(
                    "Connecte d’abord Radarr dans la barre latérale."
                    if radarr_config is None
                    else "Ajoute le film à Radarr et lance sa recherche."
                ),
            ):
                try:
                    with RadarrClient(
                        radarr_config["url"],
                        radarr_config["api_key"],
                    ) as radarr_client:
                        result = radarr_client.add_movie(
                            int(item["tmdb_id"]),
                            root_folder_path=radarr_config["root_folder_path"],
                            quality_profile_id=int(
                                radarr_config["quality_profile_id"]
                            ),
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

            if item.get("providers_ch"):
                st.caption(
                    "Disponibilité CH (JustWatch via TMDB) · "
                    + " · ".join(
                        f"{kind}: {', '.join(names)}"
                        for kind, names in item["providers_ch"].items()
                    )
                )
            with st.expander("Détail du calcul"):
                components = item.get("components", {})
                if is_safe:
                    public_columns = st.columns(3)
                    public_columns[0].metric(
                        "Note TMDB brute",
                        f"{float(item.get('vote_average') or 0):.1f}/10",
                    )
                    public_columns[1].metric(
                        "Note corrigée",
                        f"{float(item.get('bayesian_rating') or 0):.1f}/10",
                    )
                    public_columns[2].metric(
                        "Fiabilité",
                        f"{float(item.get('public_rating_reliability') or 0):.0f}%",
                    )
                    st.caption(
                        "La note publique corrigée porte 70 % du classement. "
                        "La compatibilité personnelle sert de garde-fou afin "
                        "qu’un film très bien noté mais clairement hors de tes "
                        "goûts ne soit plus présenté comme une valeur sûre."
                    )
                    if item.get("predicted_rating") is not None:
                        personal_columns = st.columns(3)
                        personal_columns[0].metric(
                            "Note prévue",
                            f"{float(item['predicted_rating']):.1f}/10",
                        )
                        personal_columns[1].metric(
                            "Envie",
                            f"{float(item.get('interest_score') or 0):.0f}/100",
                        )
                        personal_columns[2].metric(
                            "Compatibilité",
                            str(
                                item.get(
                                    "safe_eligibility_label",
                                    "—",
                                )
                            ),
                        )
                elif item.get("personal_model_used"):
                    engine_label = {
                        "islands_v07": "v0.7 · Îlots recalibrés",
                        "linear_v06": "v0.6 · Linéaire recalibré",
                        "hybrid_v08": (
                            "Challenger validé · "
                            + str(
                                item.get("personal_variant_label")
                                or "moteur hybride"
                            )
                        ),
                        "personal_v09": (
                            "v0.9 · "
                            + str(
                                item.get("personal_variant_label")
                                or "voisinage personnel"
                            )
                        ),
                    }.get(item.get("personal_engine"), "Modèle personnel")
                    st.markdown(f"**Moteur utilisé : {engine_label}**")
                    prediction_columns = st.columns(4)
                    prediction_columns[0].metric(
                        "Envie",
                        (
                            f"{float(item['interest_score']):.0f}/100"
                            if item.get("interest_score") is not None
                            else "—"
                        ),
                    )
                    prediction_columns[1].metric(
                        "Chance d’un 8+",
                        f"{float(item['like_probability']):.0f}%",
                        (
                            f"{float(item.get('like_probability_lift_points') or 0):+.0f} "
                            "points"
                        ),
                    )
                    prediction_columns[2].metric(
                        "Ta moyenne",
                        (
                            f"{float(item['user_baseline_rating']):.1f}/10"
                            if item.get("user_baseline_rating") is not None
                            else "—"
                        ),
                    )
                    prediction_columns[3].metric(
                        "Note prévue",
                        (
                            f"{float(item['predicted_rating']):.1f}/10"
                            if item.get("predicted_rating") is not None
                            else "—"
                        ),
                    )
                    st.caption(
                        "Score de classement combiné : "
                        f"{float(item.get('recommendation_score') or 0):.0f}/100 "
                        "· indice relatif de satisfaction : "
                        f"{float(item.get('satisfaction_lift_index') or 0):.0f}/100 "
                        "· confiance dans l’envie : "
                        f"{float(item.get('interest_confidence') or 0):.0f}/100."
                    )
                    st.caption(
                        "Part du voisinage local dans la prédiction : "
                        f"{100 * float(item.get('local_neighbor_weight') or 0):.0f}% "
                        "· influence directe de la note TMDB sur l’affinité : "
                        f"{100 * float(item.get('public_influence_weight') or 0):.0f}% "
                        "· TMDB reste utilisé pour la fiabilité et la qualité."
                    )
                    if item.get("personal_engine") == "islands_v07":
                        island_rows = []
                        for heading, island, polarity in (
                            (
                                "Plus proche de ce que tu apprécies",
                                item.get("positive_island"),
                                "positive",
                            ),
                            (
                                "Plus proche de ce que tu aimes moins",
                                item.get("negative_island"),
                                "negative",
                            ),
                        ):
                            if island:
                                similarity_key = (
                                    "positive_similarity"
                                    if polarity == "positive"
                                    else "negative_similarity"
                                )
                                island_rows.append(
                                    {
                                        "Repère": heading,
                                        "Îlot": island["label"],
                                        "Proximité": (
                                            f"{float(item.get(similarity_key) or 0):.0f}%"
                                        ),
                                        "Films représentatifs": ", ".join(
                                            row["title"]
                                            for row in island.get(
                                                "representatives", []
                                            )
                                        ),
                                    }
                                )
                        if island_rows:
                            st.markdown(
                                "**Voisinages retrouvés dans ton historique**"
                            )
                            st.dataframe(
                                pd.DataFrame(island_rows),
                                hide_index=True,
                                width="stretch",
                            )
                            st.caption(
                                "Écart îlot apprécié − îlot moins aimé : "
                                f"{float(item.get('island_margin') or 0):+.0f} "
                                "points de proximité."
                            )
                    else:
                        learned_rows = (
                            [
                                {
                                    "Effet appris": row["label"],
                                    "Impact estimé sur ta note": (
                                        f"{float(row['impact']):+.2f}"
                                    ),
                                }
                                for row in item.get(
                                    "learned_positive_signals", []
                                )[:5]
                            ]
                            + [
                                {
                                    "Effet appris": row["label"],
                                    "Impact estimé sur ta note": (
                                        f"{float(row['impact']):+.2f}"
                                    ),
                                }
                                for row in item.get(
                                    "learned_negative_signals", []
                                )[:5]
                            ]
                        )
                        if learned_rows:
                            st.markdown(
                                "**Effets retrouvés dans ton historique**"
                            )
                            st.dataframe(
                                pd.DataFrame(learned_rows),
                                hide_index=True,
                                width="stretch",
                            )
                    st.caption(
                        f"Couverture du modèle pour cette fiche : "
                        f"{float(item.get('model_coverage') or 0):.0f}/100. "
                        "La fourchette s’élargit lorsque les informations "
                        "sont moins proches de ton historique."
                    )
                else:
                    component_rows = [
                        ("Genres", "10 %", components.get("genres")),
                        ("Thèmes", "20 %", components.get("keywords")),
                        ("Équipe confirmée", "15 %", components.get("people")),
                        ("Histoires proches", "40 %", components.get("semantic")),
                        ("Tes corrections", "15 %", components.get("explicit")),
                    ]
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Signal personnel": label,
                                    "Poids": weight,
                                    "Indice": (
                                        round(float(value) * 100)
                                        if value is not None
                                        else "Données insuffisantes"
                                    ),
                                }
                                for label, weight, value in component_rows
                            ]
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                if item.get("semantic_neighbors"):
                    st.markdown("**Films aux histoires les plus proches**")
                    st.dataframe(
                        pd.DataFrame(item["semantic_neighbors"]),
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "title": "Film",
                            "rating": "Ta note",
                            "feedback": "Retour",
                            "sentiment": "Signal",
                            "similarity": "Proximité %",
                        },
                    )
                if item.get("matched_details"):
                    st.markdown("**Personnes et thèmes reconnus**")
                    st.dataframe(
                        pd.DataFrame(item["matched_details"]),
                        hide_index=True,
                        width="stretch",
                    )
                if not is_safe:
                    st.caption(
                        "Ordre conseillé uniquement · "
                        f"qualité publique {components.get('quality', 0) * 100:.0f}/100 "
                        f"· récence {components.get('freshness', 0) * 100:.0f}/100 "
                        f"· nouveauté {components.get('novelty', 0) * 100:.0f}/100"
                    )
                    st.caption(
                        "L’indice d’envie n’est pas une probabilité. La chance "
                        "d’un 8+ mesure la satisfaction après visionnage."
                    )
