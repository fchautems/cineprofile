from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import streamlit as st

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
    st.subheader("Réglages")
    st.write(
        "Les connexions, les mises à jour IMDb et les outils de maintenance "
        "sont regroupés ici pour laisser Suggestions et Ma liste légères."
    )

    st.markdown("### Connexions")
    render_connections(environment_path)

    st.divider()
    st.markdown("### IMDb et données")
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

    st.divider()
    st.markdown("### Personnalisation avancée")
    st.caption(
        "Ces corrections restent optionnelles : les prochaines évolutions "
        "privilégieront les retours directs sur les suggestions."
    )
    render_preferences_tab(database, profile)

    st.divider()
    show_maintenance = st.toggle(
        "Afficher la maintenance et les diagnostics",
        value=False,
        help="Audit du moteur, recalcul forcé, exports et exploration complète.",
    )
    if show_maintenance:
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
        st.divider()
        render_vivier_audit_panel(
            database,
            token=token,
            language=language,
            region=region,
            rated_count=int(counts["total"]),
            logger=logger,
        )
        st.divider()
        render_catalog_tab(database)

    return counts, profile
