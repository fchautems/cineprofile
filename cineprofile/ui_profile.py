from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from cineprofile import hybrid_model as hm
from cineprofile.profile import build_profile, render_report
from cineprofile.ui_audit import render_audit_panel
from cineprofile.ui_common import latest_profile


def _render_profile_overview(profile: dict, counts: dict[str, int]) -> None:
    st.subheader("Mon profil")
    st.write(
        "Une lecture simple de ce que ton historique IMDb révèle. Les détails "
        "techniques et les audits restent disponibles dans Réglages."
    )
    summary = profile["summary"]
    metrics = st.columns(4)
    metrics[0].metric("Films notés", f"{counts['total']:,}".replace(",", "’"))
    metrics[1].metric("Films enrichis", f"{counts['enriched']:,}".replace(",", "’"))
    metrics[2].metric("Note moyenne", f"{float(summary['average_rating']):.2f}")
    metrics[3].metric("Films notés 8+", f"{100 * float(summary['share_8_or_more']):.0f}%")

    distribution = pd.DataFrame(profile["rating_distribution"])
    figure = px.bar(
        distribution,
        x="rating",
        y="count",
        title="Distribution de tes notes",
        labels={"rating": "Note", "count": "Nombre de films"},
        color_discrete_sequence=["#c44a34"],
    )
    figure.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, width="stretch")

    st.markdown("### Repères de goût")
    dimensions = profile.get("dimensions", {})
    columns = st.columns(3)
    for column, (label, key) in zip(
        columns,
        (
            ("Genres", "genres"),
            ("Réalisateurs", "directors"),
            ("Thèmes", "keywords"),
        ),
        strict=True,
    ):
        rows = [
            row
            for row in dimensions.get(key, [])
            if float(row.get("affinity") or 0.0) > 0
        ][:5]
        column.markdown(f"**{label}**")
        if rows:
            column.write(" · ".join(str(row["name"]) for row in rows))
        else:
            column.caption("Pas encore assez de données.")

    model_summary = profile.get("personal_model", {})
    if model_summary.get("status") == "ready":
        st.caption(
            "Le moteur personnel est prêt et se met à jour après un import IMDb "
            "qui ajoute ou modifie réellement des notes."
        )


def render_profile_tab(
    database: str | Path,
    counts: dict[str, int],
    profile: dict | None,
    *,
    logger: logging.Logger,
    advanced: bool = False,
) -> dict | None:
    if not advanced:
        profile = profile or latest_profile(database)
        if profile:
            _render_profile_overview(profile, counts)
        else:
            st.info("Importe le CSV IMDb pour créer ton profil.")
        return profile

    st.subheader("Une empreinte, pas une simple liste de genres")
    st.write(
        "Le score d’affinité mesure l’écart à ta note moyenne, corrige l’effet "
        "« film universellement apprécié » et réduit les conclusions fondées "
        "sur trop peu d’exemples."
    )
    if st.button("Recalculer le profil", disabled=counts["total"] == 0):
        profile = build_profile(database)
        st.success("Profil recalculé.")

    profile = profile or latest_profile(database)
    if profile:
        distribution = pd.DataFrame(profile["rating_distribution"])
        figure = px.bar(
            distribution,
            x="rating",
            y="count",
            title="Distribution de tes notes",
            labels={"rating": "Note", "count": "Nombre de titres"},
            color_discrete_sequence=["#c44a34"],
        )
        figure.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figure, width="stretch")

        st.markdown("### Validation du moteur personnel")
        model_summary = profile.get("personal_model", {})
        if model_summary.get("status") == "ready":
            linear_metrics = model_summary["linear_v06"]
            island_metrics = model_summary.get("islands_v07")
            legacy_metrics = model_summary["legacy_v05"]
            active_engine = model_summary.get("active_engine", "legacy_v05")
            hybrid_configuration = hm.active_configuration(database)
            personal_v09_enabled = (
                hybrid_configuration is not None
                and hybrid_configuration.get("engine") == "personal_v09"
            )
            hybrid_state = (
                hm.ensure_hybrid_model(database)
                if personal_v09_enabled
                else None
            )
            active_metrics = {
                "linear_v06": linear_metrics,
                "islands_v07": island_metrics,
                "legacy_v05": legacy_metrics,
            }.get(active_engine) or legacy_metrics
            active_label = {
                "islands_v07": "v0.7 · Îlots recalibrés",
                "linear_v06": "v0.6 · Linéaire recalibré",
                "legacy_v05": "v0.5 · Historique",
            }.get(active_engine, active_engine)
            if (
                personal_v09_enabled
                and hybrid_state is not None
                and hybrid_state.status == "ready"
            ):
                active_engine = "personal_v09"
                active_metrics = hybrid_state.summary
                active_label = (
                    "0.9 · "
                    + str(
                        hybrid_configuration["configuration"].get(
                            "variant_label",
                            "Challenger validé",
                        )
                    )
                )
            elif personal_v09_enabled and hybrid_state is not None:
                st.warning(
                    "Le challenger enregistré n’a pas pu être reconstruit ; "
                    "les suggestions utilisent automatiquement le moteur "
                    "stable. Détail : "
                    + str(hybrid_state.summary.get("message") or "")
                )
            st.write(
                "La v0.9 combine un voisinage local et un seul modèle global "
                "personnel. Les mesures v0.5 à v0.7 ci-dessous restent comme "
                "repères historiques ; l’audit complet compare les variantes "
                "v0.9 sur les mêmes notes masquées. Tu n’as rien à renoter et "
                "les films jamais vus ne sont pas considérés comme négatifs."
            )
            validation_columns = st.columns(4)
            validation_columns[0].metric(
                "Films utilisés",
                f"{int(model_summary['rated_count']):,}".replace(",", "’"),
            )
            validation_columns[1].metric(
                "Moteur actif",
                active_label,
            )
            validation_columns[2].metric(
                "Top 20 appréciés retrouvé",
                f"{float(active_metrics['precision_at_20']):.0%}",
            )
            validation_columns[3].metric(
                "Classement global",
                (
                    f"{100 * float(active_metrics['auc']):.0f}/100"
                    if active_metrics.get("auc") is not None
                    else "—"
                ),
            )
            if active_engine == "personal_v09":
                st.success(
                    "Le profil personnel v0.9 est actif. Il apprend directement "
                    "tes notes ; la popularité publique ne sert plus de base "
                    "à la prédiction."
                )
            elif active_engine == "islands_v07":
                st.success(
                    "La v0.7 gagne suffisamment sur les notes cachées : ses îlots "
                    "positifs et négatifs classeront les prochaines suggestions."
                )
            else:
                st.info(
                    "La v0.7 reste un challenger. CineProfile conserve le moteur "
                    "qui donne les meilleurs résultats vérifiés sur ton historique."
                )
            if active_engine == "personal_v09":
                st.caption(
                    "La note publique ne participe pas directement à "
                    "l’affinité ; elle reste seulement un garde-fou de qualité."
                )
            else:
                st.caption(model_summary.get("selection_reason", ""))
            with st.expander("Comment lire cette validation", expanded=False):
                def metric_value(
                    metrics: dict | None,
                    key: str,
                    *,
                    scale: float = 1.0,
                    suffix: str = "",
                ) -> str:
                    if not metrics or metrics.get(key) is None:
                        return "—"
                    return f"{scale * float(metrics[key]):.2f}{suffix}"

                comparison = pd.DataFrame(
                    [
                        {
                            "Mesure": "Films appréciés dans le top 20",
                            "Version 0.5": (
                                f"{float(legacy_metrics['precision_at_20']):.0%}"
                            ),
                            "Version 0.6": (
                                f"{float(linear_metrics['precision_at_20']):.0%}"
                            ),
                            "Version 0.7": (
                                f"{float(island_metrics['precision_at_20']):.0%}"
                                if island_metrics
                                else "Non disponible"
                            ),
                        },
                        {
                            "Mesure": "Qualité du classement",
                            "Version 0.5": (
                                metric_value(
                                    legacy_metrics, "auc", scale=100, suffix="/100"
                                )
                            ),
                            "Version 0.6": (
                                metric_value(
                                    linear_metrics, "auc", scale=100, suffix="/100"
                                )
                            ),
                            "Version 0.7": (
                                metric_value(
                                    island_metrics, "auc", scale=100, suffix="/100"
                                )
                            ),
                        },
                        {
                            "Mesure": "Qualité du top 50",
                            "Version 0.5": (
                                metric_value(
                                    legacy_metrics,
                                    "ndcg_at_50",
                                    scale=100,
                                    suffix="/100",
                                )
                            ),
                            "Version 0.6": (
                                metric_value(
                                    linear_metrics,
                                    "ndcg_at_50",
                                    scale=100,
                                    suffix="/100",
                                )
                            ),
                            "Version 0.7": (
                                metric_value(
                                    island_metrics,
                                    "ndcg_at_50",
                                    scale=100,
                                    suffix="/100",
                                )
                            ),
                        },
                        {
                            "Mesure": "Erreur moyenne sur la note",
                            "Version 0.5": "—",
                            "Version 0.6": (
                                metric_value(linear_metrics, "mae", suffix=" point")
                            ),
                            "Version 0.7": (
                                metric_value(island_metrics, "mae", suffix=" point")
                            ),
                        },
                        {
                            "Mesure": "Erreur des pourcentages (Brier)",
                            "Version 0.5": "—",
                            "Version 0.6": metric_value(
                                linear_metrics,
                                "brier",
                            ),
                            "Version 0.7": metric_value(
                                island_metrics,
                                "brier",
                            ),
                        },
                    ]
                )
                st.dataframe(comparison, hide_index=True, width="stretch")
                st.caption(
                    "La qualité du classement vaut 50/100 pour un classement "
                    "sans pouvoir discriminant et 100/100 pour une séparation "
                    "parfaite des notes élevées et basses. Pour l’erreur, le plus "
                    "petit nombre est le meilleur. Le score de Brier contrôle "
                    "directement si les pourcentages annoncés correspondent aux "
                    "fréquences réellement observées."
                )
                calibration_bins = active_metrics.get("calibration_bins", [])
                if calibration_bins:
                    st.markdown(
                        "**Vérification des pourcentages du moteur actif**"
                    )
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Pourcentage annoncé": row["range"],
                                    "Films testés": int(row["count"]),
                                    "Moyenne annoncée": (
                                        f"{float(row['predicted_rate']):.0%}"
                                    ),
                                    "Films réellement appréciés": (
                                        f"{float(row['observed_rate']):.0%}"
                                    ),
                                }
                                for row in calibration_bins
                            ]
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                temporal_v06 = model_summary.get("temporal_v06")
                temporal_v07 = model_summary.get("temporal_v07")
                if temporal_v06 or temporal_v07:
                    temporal_rows = []
                    for label, temporal in (
                        ("Version 0.6", temporal_v06),
                        ("Version 0.7", temporal_v07),
                    ):
                        if temporal:
                            temporal_rows.append(
                                {
                                    "Moteur": label,
                                    "Notes récentes testées": int(
                                        temporal["tested_titles"]
                                    ),
                                    "Classement global": metric_value(
                                        temporal,
                                        "auc",
                                        scale=100,
                                        suffix="/100",
                                    ),
                                    "Erreur moyenne": metric_value(
                                        temporal,
                                        "mae",
                                        suffix=" point",
                                    ),
                                }
                            )
                    st.markdown("**Contrôle sur tes notes les plus récentes**")
                    st.dataframe(
                        pd.DataFrame(temporal_rows),
                        hide_index=True,
                        width="stretch",
                    )
            islands = model_summary.get("islands")
            if islands:
                with st.expander(
                    "Voir les îlots de goûts détectés par la v0.7",
                    expanded=False,
                ):
                    st.write(
                        "Un îlot regroupe une combinaison récurrente d’histoire, "
                        "de style, de genres, de thèmes et de personnes. Les îlots "
                        "moins aimés servent autant que les îlots appréciés."
                    )
                    for polarity, heading in (
                        ("positive", "Îlots appréciés (notes ≥ 8)"),
                        ("negative", "Îlots moins aimés (notes ≤ 6)"),
                    ):
                        rows = []
                        for island in islands.get(polarity, []):
                            representatives = ", ".join(
                                f"{row['title']} ({row['rating']:g}/10)"
                                for row in island.get("representatives", [])
                            )
                            rows.append(
                                {
                                    "Îlot": island["label"],
                                    "Films": island["size"],
                                    "Note moyenne": island["average_rating"],
                                    "Exemples": representatives,
                                }
                            )
                        st.markdown(f"**{heading}**")
                        st.dataframe(
                            pd.DataFrame(rows),
                            hide_index=True,
                            width="stretch",
                        )
        else:
            st.info(
                model_summary.get(
                    "message",
                    "Le modèle personnel sera construit automatiquement à partir "
                    "des notes déjà importées.",
                )
            )

        render_audit_panel(
            database,
            rated_count=counts["total"],
            logger=logger,
        )
        dimensions = profile["dimensions"]
        choices = {
            "Genres": "genres",
            "Réalisateurs": "directors",
            "Scénaristes": "writers",
            "Acteurs": "actors",
            "Thèmes": "keywords",
            "Directeurs photo": "cinematographers",
            "Compositeurs": "composers",
            "Décennies": "decades",
        }
        selected = st.selectbox("Dimension à explorer", list(choices))
        filter_left, filter_right = st.columns(2)
        minimum_seen = filter_left.slider(
            "Nombre minimal de films",
            1,
            20,
            2,
            help="Écarte les conclusions reposant sur trop peu de titres.",
        )
        direction = filter_right.radio(
            "Afficher",
            ["Affinités", "Rejets", "Les deux"],
            horizontal=True,
        )
        rows = [
            row
            for row in dimensions.get(choices[selected], [])
            if int(row.get("seen", 0)) >= minimum_seen
        ]
        if direction == "Affinités":
            rows = [row for row in rows if float(row.get("affinity", 0)) >= 0]
        elif direction == "Rejets":
            rows = sorted(
                [row for row in rows if float(row.get("affinity", 0)) < 0],
                key=lambda row: row["affinity"],
            )
        if rows:
            frame = pd.DataFrame(rows[:40])
            display_columns = [
                column
                for column in (
                    "name",
                    "seen",
                    "average_user_rating",
                    "average_residual_vs_public",
                    "confidence",
                    "affinity",
                )
                if column in frame.columns
            ]
            st.dataframe(
                frame[display_columns],
                width="stretch",
                hide_index=True,
            )
        else:
            if choices[selected] == "keywords":
                if dimensions.get("keywords"):
                    st.info(
                        "Aucun thème ne passe le filtre choisi. "
                        "Réduis le nombre minimal de films."
                    )
                else:
                    st.info(
                        "Aucun thème répété n’a été trouvé dans les données "
                        "enrichies."
                    )
            else:
                st.info("Cette dimension apparaîtra après l’enrichissement.")

        report_html = render_report(profile)
        col_json, col_html = st.columns(2)
        col_json.download_button(
            "Télécharger profile.json",
            json.dumps(profile, ensure_ascii=False, indent=2),
            file_name="profile.json",
            mime="application/json",
            width="stretch",
        )
        col_html.download_button(
            "Télécharger le rapport HTML",
            report_html,
            file_name="profile_report.html",
            mime="text/html",
            width="stretch",
        )
    else:
        st.info("Importe le CSV puis calcule le premier profil.")

    return profile
