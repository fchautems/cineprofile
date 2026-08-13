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
from cineprofile.radarr import RadarrClient
from cineprofile.settings import (
    forget_radarr_connection_file,
    forget_tmdb_token_file,
    save_radarr_connection_file,
    save_tmdb_token_file,
)
from cineprofile.ui_catalog import clear_catalog_cache, render_catalog_tab
from cineprofile.ui_common import latest_profile, metric_row
from cineprofile.ui_import import render_import_tab
from cineprofile.ui_my_movies import render_my_movies_tab
from cineprofile.ui_preferences import render_preferences_tab
from cineprofile.ui_profile import render_profile_tab
from cineprofile.ui_recommendations import render_recommendations_tab


ENV_PATH = Path(".env")
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

def save_tmdb_token() -> None:
    value = st.session_state.get("tmdb_token_input", "").strip()
    if not value:
        st.session_state["tmdb_token_notice"] = (
            "Ajoute d’abord un jeton avant de le mémoriser."
        )
        return
    save_tmdb_token_file(ENV_PATH, value)
    os.environ["TMDB_TOKEN"] = value
    st.session_state["tmdb_token_notice"] = "Jeton mémorisé sur ce PC."


def forget_tmdb_token() -> None:
    if ENV_PATH.exists():
        forget_tmdb_token_file(ENV_PATH)
    os.environ.pop("TMDB_TOKEN", None)
    st.session_state["tmdb_token_input"] = ""
    st.session_state["tmdb_token_notice"] = "Jeton local supprimé."


def forget_radarr_connection() -> None:
    forget_radarr_connection_file(ENV_PATH)
    os.environ.pop("RADARR_URL", None)
    os.environ.pop("RADARR_API_KEY", None)
    for state_key in (
        "radarr_url_input",
        "radarr_api_key_input",
        "radarr_root_folders",
        "radarr_quality_profiles",
        "radarr_version",
        "radarr_connection_error",
    ):
        st.session_state.pop(state_key, None)


token = st.sidebar.text_input(
    "Jeton de lecture TMDB",
    value=os.getenv("TMDB_TOKEN", ""),
    type="password",
    key="tmdb_token_input",
    help="Nécessaire seulement pour l’enrichissement et les suggestions.",
)
token_actions = st.sidebar.columns(2)
token_actions[0].button(
    "Mémoriser sur ce PC",
    on_click=save_tmdb_token,
    width="stretch",
)
token_actions[1].button(
    "Oublier",
    on_click=forget_tmdb_token,
    width="stretch",
)
token_notice = st.session_state.pop("tmdb_token_notice", None)
if token_notice:
    st.sidebar.info(token_notice)
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

radarr_config = None
with st.sidebar.expander("Radarr", expanded=False):
    radarr_url = st.text_input(
        "Adresse de Radarr",
        value=os.getenv("RADARR_URL", "http://localhost:7878"),
        key="radarr_url_input",
        placeholder="http://192.168.1.10:7878",
    )
    radarr_api_key = st.text_input(
        "Clé API Radarr",
        value=os.getenv("RADARR_API_KEY", ""),
        type="password",
        key="radarr_api_key_input",
    )
    radarr_actions = st.columns(2)
    if radarr_actions[0].button(
        "Connecter",
        key="connect_radarr",
        width="stretch",
    ):
        try:
            with RadarrClient(radarr_url, radarr_api_key) as radarr_client:
                radarr_status = radarr_client.status()
                root_folders = radarr_client.root_folders()
                quality_profiles = radarr_client.quality_profiles()
            if not root_folders:
                raise ValueError("Radarr ne contient aucun dossier racine.")
            if not quality_profiles:
                raise ValueError("Radarr ne contient aucun profil de qualité.")
            save_radarr_connection_file(ENV_PATH, radarr_url, radarr_api_key)
            os.environ["RADARR_URL"] = radarr_url.strip().rstrip("/")
            os.environ["RADARR_API_KEY"] = radarr_api_key.strip()
            st.session_state["radarr_root_folders"] = root_folders
            st.session_state["radarr_quality_profiles"] = quality_profiles
            st.session_state["radarr_version"] = str(
                radarr_status.get("version") or "connecté"
            )
            st.session_state.pop("radarr_connection_error", None)
        except Exception as exc:
            st.session_state.pop("radarr_root_folders", None)
            st.session_state.pop("radarr_quality_profiles", None)
            st.session_state["radarr_connection_error"] = str(exc)
    radarr_actions[1].button(
        "Oublier",
        key="forget_radarr",
        width="stretch",
        on_click=forget_radarr_connection,
    )

    connection_error = st.session_state.get("radarr_connection_error")
    if connection_error:
        st.error(connection_error)
    root_folders = st.session_state.get("radarr_root_folders", [])
    quality_profiles = st.session_state.get("radarr_quality_profiles", [])
    if root_folders and quality_profiles:
        st.success(
            "Radarr " + str(st.session_state.get("radarr_version", "connecté"))
        )
        root_index = st.selectbox(
            "Dossier des films",
            range(len(root_folders)),
            format_func=lambda index: str(root_folders[index].get("path") or "—"),
            key="radarr_root_index",
        )
        profile_index = st.selectbox(
            "Qualité",
            range(len(quality_profiles)),
            format_func=lambda index: str(
                quality_profiles[index].get("name") or "—"
            ),
            key="radarr_quality_index",
        )
        radarr_config = {
            "url": radarr_url.strip().rstrip("/"),
            "api_key": radarr_api_key.strip(),
            "root_folder_path": root_folders[root_index]["path"],
            "quality_profile_id": int(quality_profiles[profile_index]["id"]),
        }
        st.caption(
            "Le bouton de téléchargement ajoute le film et lance sa recherche."
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
