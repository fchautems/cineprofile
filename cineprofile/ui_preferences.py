from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from cineprofile.genre_catalog import genre_scope_label
from cineprofile.preferences import (
    ADJUSTMENT_LABELS,
    LABEL_ADJUSTMENTS,
    clear_preferences,
    load_feedback,
    load_preferences,
    remove_feedback,
    save_preferences,
)
from cineprofile.watch_interest import (
    INTEREST_ADJUSTMENT_LABELS,
    INTEREST_ENTITY_TYPE,
    INTEREST_FACTOR_DEFINITIONS,
    INTEREST_LABEL_ADJUSTMENTS,
)


PREFERENCE_DIMENSIONS = {
    "Genres": "genres",
    "Thèmes": "keywords",
    "Réalisateurs": "directors",
    "Acteurs": "actors",
    "Scénaristes": "writers",
    "Directeurs photo": "cinematographers",
    "Compositeurs": "composers",
    "Monteurs": "editors",
}


def _preference_editor_rows(
    profile: dict,
    entity_type: str,
    search: str,
    current_preferences: dict,
) -> list[dict]:
    dimension_rows = profile["dimensions"].get(entity_type, [])
    if search:
        dimension_rows = [
            row
            for row in dimension_rows
            if search.casefold() in row["name"].casefold()
        ]
    else:
        dimension_rows = dimension_rows[:60]

    editor_rows = []
    for row in dimension_rows:
        entity_id = str(row["id"])
        editor_row = {
            "entity_id": entity_id,
            "Nom": row["name"],
            "Titres notés": row.get("seen", 0),
            "Note moyenne": row.get("average_user_rating"),
            "Affinité calculée": row.get("affinity"),
            "Réglage": ADJUSTMENT_LABELS[
                int(
                    current_preferences.get(
                        (entity_type, entity_id),
                        {"adjustment": 0},
                    )["adjustment"]
                )
            ],
        }
        if entity_type == "genres":
            editor_row = {
                "entity_id": entity_id,
                "Nom": editor_row["Nom"],
                "Portée": genre_scope_label(entity_id),
                **{
                    key: value
                    for key, value in editor_row.items()
                    if key not in {"entity_id", "Nom"}
                },
            }
        editor_rows.append(editor_row)
    return editor_rows


def render_preferences_tab(database: str | Path, profile: dict | None) -> None:
    st.subheader("Corriger ce que CineProfile a compris")
    st.write(
        "La fréquence ne vaut pas préférence. Tes corrections sont prioritaires "
        "et ne seront jamais écrasées par le recalcul automatique."
    )
    if not profile:
        st.info("Importe d’abord ton historique.")
        return

    current_preferences = load_preferences(database)
    with st.expander("Ce qui me donne envie de lancer un film", expanded=True):
        st.write(
            "Ces réglages agissent sur l’envie avant visionnage, jamais sur "
            "tes anciennes notes. Ils distinguent par exemple une comédie "
            "noire d’une comédie générique."
        )
        interest_rows = []
        for factor, configuration in INTEREST_FACTOR_DEFINITIONS.items():
            adjustment = int(
                current_preferences.get(
                    (INTEREST_ENTITY_TYPE, factor),
                    {"adjustment": configuration["default"]},
                )["adjustment"]
            )
            interest_rows.append(
                {
                    "factor": factor,
                    "Signal": configuration["label"],
                    "Effet": INTEREST_ADJUSTMENT_LABELS[adjustment],
                    "Rôle": configuration["description"],
                }
            )
        interest_frame = pd.DataFrame(interest_rows).set_index("factor")
        edited_interest = st.data_editor(
            interest_frame,
            hide_index=True,
            width="stretch",
            disabled=["Signal", "Rôle"],
            column_config={
                "Effet": st.column_config.SelectboxColumn(
                    "Ton réglage",
                    options=list(INTEREST_LABEL_ADJUSTMENTS),
                    required=True,
                )
            },
        )
        interest_actions = st.columns(2)
        if interest_actions[0].button(
            "Enregistrer mon profil d’envie",
            width="stretch",
        ):
            save_preferences(
                [
                    {
                        "entity_type": INTEREST_ENTITY_TYPE,
                        "entity_id": str(factor),
                        "entity_name": str(row["Signal"]),
                        "adjustment": INTEREST_LABEL_ADJUSTMENTS[row["Effet"]],
                    }
                    for factor, row in edited_interest.iterrows()
                ],
                database,
            )
            st.session_state.pop("recommendations", None)
            st.success("Profil d’envie enregistré.")
        if interest_actions[1].button(
            "Restaurer les réglages conseillés",
            width="stretch",
        ):
            clear_preferences(INTEREST_ENTITY_TYPE, database)
            st.session_state.pop("recommendations", None)
            st.rerun()

    preference_columns = st.columns([1, 2])
    preference_label = preference_columns[0].selectbox(
        "Dimension",
        list(PREFERENCE_DIMENSIONS),
    )
    preference_search = preference_columns[1].text_input(
        "Chercher un élément",
        placeholder="science-fiction, biographie, Nolan…",
    )
    entity_type = PREFERENCE_DIMENSIONS[preference_label]
    editor_rows = _preference_editor_rows(
        profile,
        entity_type,
        preference_search,
        current_preferences,
    )

    if editor_rows:
        editor_frame = pd.DataFrame(editor_rows).set_index("entity_id")
        edited_preferences = st.data_editor(
            editor_frame,
            hide_index=True,
            width="stretch",
            disabled=(
                [
                    "Nom",
                    "Titres notés",
                    "Note moyenne",
                    "Affinité calculée",
                ]
                + (["Portée"] if entity_type == "genres" else [])
            ),
            column_config={
                "Réglage": st.column_config.SelectboxColumn(
                    "Ta correction",
                    options=list(LABEL_ADJUSTMENTS),
                    required=True,
                )
            },
        )
        save_columns = st.columns(2)
        if save_columns[0].button(
            "Enregistrer ces corrections",
            width="stretch",
        ):
            save_preferences(
                [
                    {
                        "entity_type": entity_type,
                        "entity_id": str(entity_id),
                        "entity_name": row["Nom"],
                        "adjustment": LABEL_ADJUSTMENTS[row["Réglage"]],
                    }
                    for entity_id, row in edited_preferences.iterrows()
                ],
                database,
            )
            st.session_state.pop("recommendations", None)
            st.success("Corrections enregistrées.")
        if save_columns[1].button(
            "Remettre cette liste en automatique",
            width="stretch",
        ):
            save_preferences(
                [
                    {
                        "entity_type": entity_type,
                        "entity_id": str(entity_id),
                        "entity_name": row["Nom"],
                        "adjustment": 0,
                    }
                    for entity_id, row in edited_preferences.iterrows()
                ],
                database,
            )
            st.session_state.pop("recommendations", None)
            st.rerun()
    else:
        st.info("Aucun élément correspondant dans le profil.")

    st.divider()
    st.markdown("#### Tes retours sur les suggestions")
    feedback_rows = list(load_feedback(database).values())
    if not feedback_rows:
        st.caption("Aucun retour enregistré pour le moment.")
        return

    st.dataframe(
        pd.DataFrame(feedback_rows)[["title", "action", "updated_at"]],
        hide_index=True,
        width="stretch",
        column_config={
            "title": "Film",
            "action": "Retour",
            "updated_at": "Date",
        },
    )
    feedback_labels = {
        f"{row['title']} · {row['action']}": row["tmdb_id"]
        for row in feedback_rows
    }
    feedback_to_remove = st.selectbox(
        "Annuler un retour",
        ["—"] + list(feedback_labels),
    )
    if (
        feedback_to_remove != "—"
        and st.button("Annuler le retour sélectionné")
    ):
        remove_feedback(feedback_labels[feedback_to_remove], database)
        st.rerun()
