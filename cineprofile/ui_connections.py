from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st

from cineprofile.radarr import RadarrClient
from cineprofile.settings import (
    CONNECTION_KEYS,
    forget_radarr_connection_file,
    forget_tmdb_token_file,
    read_connection_settings,
    save_connection_settings,
)


def _radarr_signature(url: str, api_key: str) -> str:
    return hashlib.sha256(f"{url}\0{api_key}".encode()).hexdigest()


def _connect_radarr(url: str, api_key: str) -> tuple[dict, list[dict], list[dict]]:
    with RadarrClient(url, api_key) as client:
        status = client.status()
        root_folders = client.root_folders()
        quality_profiles = client.quality_profiles()
    if not root_folders:
        raise ValueError("Radarr ne contient aucun dossier racine.")
    if not quality_profiles:
        raise ValueError("Radarr ne contient aucun profil de qualité.")
    return status, root_folders, quality_profiles


def _remember_radarr_session(
    url: str,
    api_key: str,
    status: dict,
    root_folders: list[dict],
    quality_profiles: list[dict],
) -> None:
    st.session_state["radarr_connection_signature"] = _radarr_signature(
        url,
        api_key,
    )
    st.session_state["radarr_root_folders"] = root_folders
    st.session_state["radarr_quality_profiles"] = quality_profiles
    st.session_state["radarr_version"] = str(status.get("version") or "connecté")
    st.session_state.pop("radarr_connection_error", None)


def _forget_radarr_session() -> None:
    for key in (
        "radarr_connection_signature",
        "radarr_root_folders",
        "radarr_quality_profiles",
        "radarr_version",
        "radarr_connection_error",
    ):
        st.session_state.pop(key, None)


def _select_index(rows: list[dict], field: str, expected: str) -> int:
    return next(
        (
            index
            for index, row in enumerate(rows)
            if str(row.get(field) or "") == str(expected or "")
        ),
        0,
    )


def _with_environment_fallback(stored: dict[str, str]) -> dict[str, str]:
    return {
        key: stored.get(key, "") or os.getenv(key, "").strip()
        for key in CONNECTION_KEYS
    }


def render_connections_sidebar(
    environment_path: str | Path,
) -> tuple[str, dict | None]:
    """Render compact persistent connection settings and return active values."""
    try:
        stored = read_connection_settings(environment_path)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        return "", None

    settings = _with_environment_fallback(stored)
    token = settings["TMDB_TOKEN"]
    radarr_url = settings["RADARR_URL"]
    radarr_key = settings["RADARR_API_KEY"]
    signature = _radarr_signature(radarr_url, radarr_key) if radarr_key else ""

    if (
        radarr_url
        and radarr_key
        and st.session_state.get("radarr_connection_signature") != signature
    ):
        try:
            status, roots, profiles = _connect_radarr(radarr_url, radarr_key)
        except Exception as exc:
            _forget_radarr_session()
            st.session_state["radarr_connection_signature"] = signature
            st.session_state["radarr_connection_error"] = str(exc)
        else:
            _remember_radarr_session(
                radarr_url,
                radarr_key,
                status,
                roots,
                profiles,
            )

    configured = bool(token)
    editing = st.session_state.get("connection_edit_mode", not configured)
    roots = st.session_state.get("radarr_root_folders", [])
    profiles = st.session_state.get("radarr_quality_profiles", [])

    with st.sidebar.expander("Connexions", expanded=editing or not configured):
        if not editing:
            st.success("TMDB configuré")
            if roots and profiles:
                st.success(
                    "Radarr connecté · v"
                    + str(st.session_state.get("radarr_version", "—"))
                )
            elif radarr_key:
                st.warning("Radarr configuré mais actuellement injoignable")
                if st.session_state.get("radarr_connection_error"):
                    st.caption(str(st.session_state["radarr_connection_error"]))
            else:
                st.caption("Radarr non configuré")
            if st.button("Modifier les connexions", width="stretch"):
                st.session_state["connection_edit_mode"] = True
                st.rerun()
        else:
            tmdb_input = st.text_input(
                "Clé TMDB",
                value=token,
                type="password",
                help="Elle sera enregistrée localement dans le fichier .env.",
            )
            radarr_url_input = st.text_input(
                "Adresse de Radarr",
                value=radarr_url or "http://localhost:7878",
                placeholder="http://192.168.1.10:7878",
            )
            radarr_key_input = st.text_input(
                "Clé API Radarr",
                value=radarr_key,
                type="password",
            )

            selected_root = settings["RADARR_ROOT_FOLDER"]
            selected_profile_id = settings["RADARR_QUALITY_PROFILE_ID"]
            if roots and profiles:
                root_index = st.selectbox(
                    "Dossier des films",
                    range(len(roots)),
                    index=_select_index(roots, "path", selected_root),
                    format_func=lambda index: str(roots[index].get("path") or "—"),
                )
                profile_index = st.selectbox(
                    "Profil de qualité",
                    range(len(profiles)),
                    index=_select_index(profiles, "id", selected_profile_id),
                    format_func=lambda index: str(
                        profiles[index].get("name") or "—"
                    ),
                )
                selected_root = str(roots[root_index].get("path") or "")
                selected_profile_id = str(profiles[profile_index].get("id") or "")

            if st.button("Enregistrer", type="primary", width="stretch"):
                clean_url = radarr_url_input.strip().rstrip("/")
                clean_key = radarr_key_input.strip()
                try:
                    if clean_url and clean_key:
                        status, new_roots, new_profiles = _connect_radarr(
                            clean_url,
                            clean_key,
                        )
                        valid_roots = {
                            str(row.get("path") or "") for row in new_roots
                        }
                        valid_profiles = {
                            str(row.get("id") or "") for row in new_profiles
                        }
                        if selected_root not in valid_roots:
                            selected_root = str(new_roots[0].get("path") or "")
                        if selected_profile_id not in valid_profiles:
                            selected_profile_id = str(
                                new_profiles[0].get("id") or ""
                            )
                    else:
                        status, new_roots, new_profiles = {}, [], []
                    save_connection_settings(
                        environment_path,
                        tmdb_token=tmdb_input,
                        radarr_url=clean_url,
                        radarr_api_key=clean_key,
                        radarr_root_folder=selected_root,
                        radarr_quality_profile_id=selected_profile_id,
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    os.environ["TMDB_TOKEN"] = tmdb_input.strip()
                    if clean_key:
                        os.environ["RADARR_URL"] = clean_url
                        os.environ["RADARR_API_KEY"] = clean_key
                        _remember_radarr_session(
                            clean_url,
                            clean_key,
                            status,
                            new_roots,
                            new_profiles,
                        )
                    else:
                        os.environ.pop("RADARR_URL", None)
                        os.environ.pop("RADARR_API_KEY", None)
                        _forget_radarr_session()
                    st.session_state["connection_edit_mode"] = False
                    st.session_state["connection_notice"] = (
                        "Connexions enregistrées sur ce PC."
                    )
                    st.rerun()

            if configured and st.button("Annuler", width="stretch"):
                st.session_state["connection_edit_mode"] = False
                st.rerun()

            if configured or radarr_key:
                st.caption("Supprimer une connexion enregistrée")
                delete_columns = st.columns(2)
                if delete_columns[0].button("Oublier TMDB"):
                    forget_tmdb_token_file(environment_path)
                    os.environ.pop("TMDB_TOKEN", None)
                    st.session_state["connection_edit_mode"] = True
                    st.rerun()
                if delete_columns[1].button("Oublier Radarr"):
                    forget_radarr_connection_file(environment_path)
                    os.environ.pop("RADARR_URL", None)
                    os.environ.pop("RADARR_API_KEY", None)
                    _forget_radarr_session()
                    st.rerun()

    notice = st.session_state.pop("connection_notice", None)
    if notice:
        st.sidebar.success(notice)

    # Re-read after UI callbacks so the returned configuration always comes
    # from durable storage rather than from password widget state.
    try:
        stored = read_connection_settings(environment_path)
    except ValueError:
        return "", None
    settings = _with_environment_fallback(stored)
    token = settings["TMDB_TOKEN"]
    roots = st.session_state.get("radarr_root_folders", [])
    profiles = st.session_state.get("radarr_quality_profiles", [])
    if not (
        settings["RADARR_URL"]
        and settings["RADARR_API_KEY"]
        and roots
        and profiles
    ):
        return token, None
    root_index = _select_index(roots, "path", settings["RADARR_ROOT_FOLDER"])
    profile_index = _select_index(
        profiles,
        "id",
        settings["RADARR_QUALITY_PROFILE_ID"],
    )
    return token, {
        "url": settings["RADARR_URL"],
        "api_key": settings["RADARR_API_KEY"],
        "root_folder_path": str(roots[root_index]["path"]),
        "quality_profile_id": int(profiles[profile_index]["id"]),
    }
