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
from cineprofile.ui_catalog import clear_catalog_cache
from cineprofile.ui_common import latest_profile
from cineprofile.ui_connections import (
    render_connection_status_sidebar,
    resolve_connections,
)
from cineprofile.ui_my_movies import render_my_movies_tab
from cineprofile.ui_profile import render_profile_tab
from cineprofile.ui_recommendations import render_recommendations_tab
from cineprofile.ui_settings import render_settings_tab


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
      [class*="st-key-watchlist_"] button[kind="primary"] {
        background:#238636; border-color:#238636; color:white;
      }
      [class*="st-key-not_interested_"] button[kind="primary"] {
        background:#cf222e; border-color:#cf222e; color:white;
      }
      [class*="st-key-already_seen_"] button[kind="primary"] {
        background:#0969da; border-color:#0969da; color:white;
      }
      [class*="st-key-radarr_"] button[kind="primary"] {
        background:#8250df; border-color:#8250df; color:white;
      }
      [class*="st-key-watchlist_"] button,
      [class*="st-key-not_interested_"] button,
      [class*="st-key-already_seen_"] button,
      [class*="st-key-radarr_"] button {
        min-height:2.6rem; border-radius:999px; font-weight:650;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="cp-kicker">Profil local & explicable</div>', unsafe_allow_html=True)
st.title("CineProfile")
st.markdown(
    """
    <div class="cp-intro">
    Découvre des films, garde ceux qui comptent dans ta liste et envoie-les à
    Radarr quand tu veux les voir.
    </div>
    """,
    unsafe_allow_html=True,
)

token, radarr_config = resolve_connections(ENV_PATH)
render_connection_status_sidebar(token, radarr_config)
st.sidebar.caption(f"Version : {__version__}")

counts = database_counts(DB_PATH)
profile = latest_profile(DB_PATH)
if counts["total"] and profile_needs_refresh(profile, counts):
    profile = build_profile(DB_PATH)
(
    tab_recommend,
    tab_my_movies,
    tab_profile,
    tab_preferences,
) = st.tabs(
    [
        "Suggestions",
        "Ma liste",
        "Mon profil",
        "Réglages",
    ],
    default="Réglages" if not token else "Suggestions",
    key="main_navigation",
    on_change="rerun",
)
current_main_tab = str(st.session_state.get("main_navigation") or "")
previous_main_tab = st.session_state.get("previous_main_navigation")
refresh_my_list_on_open = (
    current_main_tab == "Ma liste" and previous_main_tab != "Ma liste"
)
st.session_state["previous_main_navigation"] = current_main_tab

with tab_recommend:
    if tab_recommend.open:
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
    if tab_my_movies.open:
        render_my_movies_tab(
            DB_PATH,
            radarr_config=radarr_config,
            refresh_on_open=refresh_my_list_on_open,
        )
with tab_profile:
    if tab_profile.open:
        profile = render_profile_tab(
            DB_PATH,
            counts,
            profile,
            logger=APP_LOGGER,
        )
with tab_preferences:
    if tab_preferences.open:
        counts, profile = render_settings_tab(
            DB_PATH,
            ENV_PATH,
            token=token,
            language=LANGUAGE,
            region=REGION,
            counts=counts,
            profile=profile,
            clear_catalog_cache=clear_catalog_cache,
            logger=APP_LOGGER,
        )
