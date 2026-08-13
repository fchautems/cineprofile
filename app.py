from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import cineprofile.recommender as recommender_module
from cineprofile import __version__
from cineprofile.compat import (
    CineProfileVersionMismatch,
    ensure_recommendation_protocol,
)
from cineprofile.db import initialize
from cineprofile.diagnostics import configure_logging
from cineprofile.imdb_import import database_counts
from cineprofile.profile import build_profile, profile_needs_refresh
from cineprofile.ui_catalog import clear_catalog_cache, render_catalog_tab
from cineprofile.ui_common import latest_profile, metric_row
from cineprofile.ui_connections import render_connections_sidebar
from cineprofile.ui_import import render_import_tab
from cineprofile.ui_my_movies import render_my_movies_tab
from cineprofile.ui_preferences import render_preferences_tab
from cineprofile.ui_profile import render_profile_tab
from cineprofile.ui_recommendations import render_recommendations_tab


ENV_PATH = Path(".env")
if ENV_PATH.is_file():
    load_dotenv(ENV_PATH)
DB_PATH = Path(os.getenv("CINEPROFILE_DB", "data/cineprofile.db"))
LANGUAGE = os.getenv("CINEPROFILE_LANGUAGE", "fr-FR")
REGION = os.getenv("CINEPROFILE_REGION", "CH")

st.set_page_config(
    page_title="CineProfile",
    page_icon="🎞️",
    layout="wide",
)
APP_LOGGER = configure_logging(DB_PATH)
try:
    initialize(DB_PATH)
except Exception as exc:
    APP_LOGGER.exception("database_initialization_failed")
    st.error(
        "La base CineProfile n’a pas pu être ouverte. Aucune donnée n’a été "
        f"modifiée. Détail : {exc}"
    )
    st.stop()

try:
    ensure_recommendation_protocol(recommender_module)
except CineProfileVersionMismatch as exc:
    st.error(str(exc))
    st.stop()

if (
    st.session_state.get("recommendation_ui_protocol")
    != recommender_module.RECOMMENDATION_PROTOCOL
):
    st.session_state.pop("recommendations", None)
    st.session_state.pop("recommendation_lists", None)
    st.session_state.pop("recommendation_diagnostics", None)
    st.session_state["recommendation_ui_protocol"] = (
        recommender_module.RECOMMENDATION_PROTOCOL
    )

st.markdown(
    """
    <style>
      .block-container { max-width: 1180px; padding-top: 2.4rem; }
      h1 { letter-spacing: -.045em; }
      [data-testid="stMetric"] {
        background: #fffdf8; border: 1px solid #e4dfd2;
        border-radius: 16px; padding: 14px 16px;
      }
      .cp-kicker { color:#c44a34; text-transform:uppercase; letter-spacing:.16em;
        font-weight:750; font-size:.74rem; }
      .cp-intro { color:#66685f; max-width:780px; font-size:1.08rem;
        line-height:1.58; margin-bottom:1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="cp-kicker">Profil local & explicable</div>', unsafe_allow_html=True)
st.title("CineProfile")
st.markdown(
    """
    <div class="cp-intro">
    Importe ton historique IMDb, enrichis-le avec les œuvres, les personnes et
    les thèmes, puis transforme-le en un modèle de goût durable. Chaque
    recommandation explique précisément les signaux utilisés.
    </div>
    """,
    unsafe_allow_html=True,
)

token, radarr_config = render_connections_sidebar(ENV_PATH)
st.sidebar.caption(f"Région : {REGION} · Langue : {LANGUAGE}")
st.sidebar.caption(f"Base : {DB_PATH}")
st.sidebar.caption(f"Version : {__version__}")
technical_log = DB_PATH.parent / "logs" / "cineprofile.log"
if technical_log.is_file():
    st.sidebar.download_button(
        "Télécharger le journal technique",
        data=technical_log.read_bytes(),
        file_name="cineprofile.log",
        mime="text/plain",
        width="stretch",
    )

counts = database_counts(DB_PATH)
profile = latest_profile(DB_PATH)
if counts["total"] and profile_needs_refresh(profile, counts):
    profile = build_profile(DB_PATH)
metric_row(counts, profile)

(
    tab_import,
    tab_profile,
    tab_catalog,
    tab_recommend,
    tab_my_movies,
    tab_preferences,
) = st.tabs(
    [
        "1 · Importer et enrichir",
        "2 · Comprendre le profil",
        "3 · Explorer la vidéothèque",
        "4 · Suggestions",
        "5 · Mes films",
        "6 · Ajuster le profil",
    ]
)

with tab_import:
    counts, profile = render_import_tab(
        DB_PATH,
        token=token,
        language=LANGUAGE,
        region=REGION,
        counts=counts,
        profile=profile,
        clear_catalog_cache=clear_catalog_cache,
        logger=APP_LOGGER,
    )

with tab_profile:
    profile = render_profile_tab(
        DB_PATH,
        counts,
        profile,
        logger=APP_LOGGER,
    )
with tab_catalog:
    render_catalog_tab(DB_PATH)
with tab_recommend:
    render_recommendations_tab(
        DB_PATH,
        profile,
        token=token,
        language=LANGUAGE,
        region=REGION,
        logger=APP_LOGGER,
        radarr_config=radarr_config,
    )
with tab_my_movies:
    render_my_movies_tab(DB_PATH, radarr_config=radarr_config)
with tab_preferences:
    render_preferences_tab(DB_PATH, profile)
