from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from cineprofile import __version__
from cineprofile.audit import backtest_audit_path, run_backtest_audit
from cineprofile import hybrid_model as hm


ENGINE_LABELS = {
    "public_baseline": "Note publique seule",
    "legacy_v05": "v0.5 · Indice historique",
    "linear_v06": "v0.6 · Linéaire",
    "islands_v07": "v0.7 · Îlots",
}


def _engine_label(engine: object) -> str:
    value = str(engine or "")
    if value in ENGINE_LABELS:
        return ENGINE_LABELS[value]
    if value.startswith("personal_") and value.endswith("_v09"):
        variant = value.removeprefix("personal_").removesuffix("_v09")
        configuration = hm.HYBRID_VARIANTS.get(variant)
        if configuration:
            return "v0.9 · " + str(configuration["label"])
    return value

AUDIT_CONFIGURATIONS = {
    "Complet — 5 découpages": {
        "repeats": 5,
        "learning_curve_repeats": 2,
    },
    "Très approfondi — 10 découpages": {
        "repeats": 10,
        "learning_curve_repeats": 3,
    },
}


def _average(metrics: dict, key: str) -> float | None:
    value = metrics.get(key, {}).get("mean")
    return round(float(value), 4) if value is not None else None


def _random_holdout_rows(payload: dict) -> list[dict]:
    summaries = payload.get("random_holdouts", {}).get("engine_summaries", {})
    rows = []
    for engine, metrics in summaries.items():
        precision = _average(metrics, "precision_at_20")
        rows.append(
            {
                "Moteur": _engine_label(engine),
                "Films appréciés dans le top 10": (
                    f"{100 * float(_average(metrics, 'precision_at_10')):.1f} %"
                    if _average(metrics, "precision_at_10") is not None
                    else "—"
                ),
                "Films appréciés dans le top 20": (
                    f"{100 * precision:.1f} %" if precision is not None else "—"
                ),
                "Qualité du top 20": _average(metrics, "ndcg_at_20"),
                "Qualité du top 10": _average(metrics, "ndcg_at_10"),
                "Classement global": _average(metrics, "auc"),
                "Erreur moyenne": _average(metrics, "mae"),
                "Erreur des probabilités": _average(metrics, "brier"),
                "Calibration du top 10": _average(metrics, "ece_at_10"),
            }
        )
        rows[-1]["Moteur"] = _engine_label(engine)
    return rows


def _chronological_rows(payload: dict) -> list[dict]:
    chronological = payload.get("chronological")
    if not chronological:
        return []
    rows = []
    for metrics in chronological.get("metrics", []):
        precision = metrics.get("precision_at_20")
        engine = metrics.get("engine")
        rows.append(
            {
                "Moteur": _engine_label(engine),
                "Films appréciés dans le top 10": (
                    f"{100 * float(metrics['precision_at_10']):.1f} %"
                    if metrics.get("precision_at_10") is not None
                    else "—"
                ),
                "Films appréciés dans le top 20": (
                    f"{100 * float(precision):.1f} %"
                    if precision is not None
                    else "—"
                ),
                "Qualité du top 20": metrics.get("ndcg_at_20"),
                "Qualité du top 10": metrics.get("ndcg_at_10"),
                "Classement global": metrics.get("auc"),
                "Erreur moyenne": metrics.get("mae"),
            }
        )
    return rows


def _learning_rows(payload: dict) -> list[dict]:
    rows = []
    summaries = payload.get("learning_curve", {}).get("summaries", {})
    for train_size, metrics in summaries.items():
        rows.append(
            {
                "Films d’apprentissage": int(train_size),
                "Top 20 apprécié": metrics.get("precision_at_20", {}).get("mean"),
                "Classement global": metrics.get("auc", {}).get("mean"),
                "Erreur moyenne": metrics.get("mae", {}).get("mean"),
            }
        )
    return sorted(rows, key=lambda row: row["Films d’apprentissage"])


def _load_payload(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _render_report(
    path: Path,
    payload: dict,
    database: str | Path,
) -> None:
    integrity = payload.get("integrity", {})
    if integrity.get("source_unchanged"):
        st.success(
            "Contrôle d’intégrité réussi : la base source est strictement inchangée."
        )
    elif integrity:
        st.error("Le contrôle d’intégrité de la base source a échoué.")
    if payload.get("semantic_preparation_error"):
        st.warning(
            "Le modèle sémantique profond n’a pas pu être préparé. Les "
            "variantes concernées ont été ignorées, mais les autres contrôles "
            "restent valides. Détail : "
            + str(payload["semantic_preparation_error"])
        )

    random_rows = _random_holdout_rows(payload)
    if random_rows:
        st.markdown("**Résultats moyens sur les groupes cachés**")
        st.dataframe(
            pd.DataFrame(random_rows),
            hide_index=True,
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(format="%.3f")
                for column in (
                    "Qualité du top 10",
                    "Qualité du top 20",
                    "Classement global",
                    "Erreur moyenne",
                    "Erreur des probabilités",
                    "Calibration du top 10",
                )
            },
        )

    retrieval = payload.get("candidate_retrieval", {}).get("summary", {})
    if retrieval:
        st.markdown("**Test de la récupération des candidats**")
        retrieval_columns = st.columns(4)
        retrieval_columns[0].metric(
            "Aimés retrouvés · top 20",
            f"{100 * float(_average(retrieval, 'liked_recall_at_20') or 0):.1f}%",
        )
        retrieval_columns[1].metric(
            "Aimés retrouvés · top 50",
            f"{100 * float(_average(retrieval, 'liked_recall_at_50') or 0):.1f}%",
        )
        retrieval_columns[2].metric(
            "Rejets dans le top 20",
            f"{100 * float(_average(retrieval, 'disliked_share_at_20') or 0):.1f}%",
        )
        retrieval_columns[3].metric(
            "Qualité du top 20",
            f"{100 * float(_average(retrieval, 'ndcg_at_20') or 0):.1f}/100",
        )
        st.caption(
            "Ce contrôle vérifie que le moteur retrouve des films appréciés "
            "masqués avant même le classement final."
        )

    chronological_rows = _chronological_rows(payload)
    if chronological_rows:
        st.markdown("**Contrôle sur les notes les plus récentes**")
        st.dataframe(
            pd.DataFrame(chronological_rows),
            hide_index=True,
            width="stretch",
        )

    learning_rows = _learning_rows(payload)
    if learning_rows:
        st.markdown("**Courbe d’apprentissage de la v0.6**")
        st.dataframe(
            pd.DataFrame(learning_rows),
            hide_index=True,
            width="stretch",
        )

    findings = payload.get("automated_findings", {})
    optimizer = findings.get("optimizer_recommendation", {})
    if optimizer:
        if optimizer.get("decision") == "promote":
            st.success(str(optimizer.get("message") or ""))
            st.caption(
                "Le bouton choisit une seule variante globale v0.9. Le "
                "voisinage local reste indépendant et aucun score v0.6 "
                "n’est additionné."
            )
            if st.button(
                "Activer ce challenger validé",
                key="apply_audit_challenger_"
                + str(payload.get("created_at") or ""),
                type="primary",
                width="stretch",
            ):
                hm.apply_configuration(
                    database,
                    variant=str(optimizer["variant"]),
                    audit_created_at=str(payload.get("created_at") or ""),
                    selected_alpha=(
                        float(optimizer["selected_alpha"])
                        if optimizer.get("selected_alpha") is not None
                        else None
                    ),
                )
                st.success(
                    "Variante v0.9 activée. Les prochaines suggestions "
                    "utiliseront ce modèle global avec le voisinage local."
                )
        else:
            st.info(str(optimizer.get("message") or ""))
        with st.expander("Voir les critères de promotion", expanded=False):
            gate_rows = []
            for name, gate in optimizer.get("gates", {}).items():
                gate_rows.append(
                    {
                        "Contrôle": name.replace("_", " "),
                        "Résultat": "Réussi" if gate.get("passed") else "Échoué",
                        "Valeur observée": gate.get(
                            "observed",
                            gate.get("observed_ndcg"),
                        ),
                    }
                )
            if gate_rows:
                st.dataframe(
                    pd.DataFrame(gate_rows),
                    hide_index=True,
                    width="stretch",
                )

    active = hm.active_configuration(database)
    if active:
        label = active.get("configuration", {}).get(
            "variant_label",
            active.get("engine"),
        )
        if active.get("engine") == "personal_v09":
            st.success(f"Moteur personnel v0.9 actif : {label}.")
            with st.expander("Option de retour à l’ancien moteur", expanded=False):
                st.caption(
                    "À utiliser seulement pour comparer ou diagnostiquer une "
                    "régression : la v0.6 remet la note publique au centre."
                )
                if st.button(
                    "Utiliser temporairement la v0.6",
                    key="restore_linear_engine",
                    width="stretch",
                ):
                    hm.restore_linear_configuration(database)
                    st.success("La v0.6 linéaire est temporairement active.")
        else:
            st.warning(
                "Ancien moteur v0.6 actif : la recherche personnelle v0.9 "
                "n’est pas utilisée."
            )
            if st.button(
                "Réactiver la v0.9 personnelle",
                key="restore_personal_v09",
                width="stretch",
                type="primary",
            ):
                hm.apply_configuration(
                    database,
                    variant=hm.DEFAULT_PERSONAL_VARIANT,
                    audit_created_at=None,
                )
                st.success("La configuration personnelle v0.9 est réactivée.")
    interpretation = findings.get("learning_curve", {}).get("interpretation")
    if interpretation:
        st.info(interpretation)
    if findings.get("public_baseline_materially_better_at_top20"):
        st.warning(
            "Alerte importante : sur les groupes cachés, la note publique seule "
            "bat nettement les moteurs personnels dans le top 20. Le moteur actif "
            "ne doit pas être modifié avant analyse du rapport."
        )
    for warning in findings.get("warnings", []):
        if warning.get("code") != "public_baseline_beats_personal_top20":
            st.warning(str(warning.get("message") or warning))

    st.download_button(
        "Télécharger le rapport d’audit",
        data=path.read_bytes(),
        file_name=path.name,
        mime="application/json",
        width="stretch",
    )
    st.caption(
        "Ce fichier peut être joint directement ici. Il ne contient ni jeton "
        "TMDB, ni résumés complets, ni historique intégral."
    )


def render_audit_panel(
    database: str | Path,
    *,
    rated_count: int,
    logger: logging.Logger,
) -> None:
    with st.expander(
        "Audit complet du moteur — sans modifier le profil",
        expanded=False,
    ):
        st.write(
            "Cet audit travaille sur une copie temporaire de la base. Il "
            "compare séparément la note publique, la v0.5, la v0.6, la v0.7 "
            "et six variantes v0.9 (métadonnées, syntaxe et sémantique) sur des "
            "films entièrement cachés, puis contrôle les notes les plus "
            "récentes. Il contrôle aussi si la récupération sémantique retrouve "
            "les films appréciés masqués. Tes notes, enrichissements, "
            "préférences et recommandations actives ne sont jamais modifiés. "
            "Le cache local du modèle sémantique peut être créé au premier "
            "lancement."
        )
        audit_mode = st.selectbox(
            "Profondeur de l’audit",
            list(AUDIT_CONFIGURATIONS),
            help=(
                "Les deux modes testent plusieurs tailles d’apprentissage "
                "et un découpage chronologique. Le second réduit davantage "
                "l’effet du hasard, mais prend plus de temps."
            ),
        )
        configuration = AUDIT_CONFIGURATIONS[audit_mode]
        if st.button(
            "Lancer l’audit du moteur",
            disabled=rated_count < 120,
            width="stretch",
            type="primary",
        ):
            progress_bar = st.progress(0.0)
            progress_message = st.empty()

            def on_progress(completed: int, total: int, message: str) -> None:
                progress_bar.progress(min(1.0, completed / max(1, total)))
                progress_message.caption(
                    f"{message} · étape {completed + 1}/{max(1, total)}"
                )

            try:
                payload = run_backtest_audit(
                    database,
                    repeats=configuration["repeats"],
                    learning_curve_repeats=configuration[
                        "learning_curve_repeats"
                    ],
                    on_progress=on_progress,
                )
                path = backtest_audit_path(database, payload)
                if not path.is_file():
                    raise OSError("le rapport final n’a pas été enregistré")
            except Exception as exc:
                logger.exception("audit_failed | app_version=%s", __version__)
                st.error(f"L’audit n’a pas pu se terminer : {exc}")
            else:
                progress_bar.progress(1.0)
                progress_message.caption("Audit terminé.")
                st.session_state["latest_audit_path"] = str(path)
                st.success(
                    "Audit terminé. Aucun moteur n’est activé automatiquement."
                )

        path_value = st.session_state.get("latest_audit_path")
        if not path_value:
            return
        path = Path(str(path_value))
        if not path.is_file():
            st.warning("Le dernier rapport d’audit n’est plus disponible.")
            return
        payload = _load_payload(path)
        if payload is None:
            st.error("Le dernier rapport d’audit est illisible.")
            return
        _render_report(path, payload, database)
