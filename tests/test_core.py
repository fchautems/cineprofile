from __future__ import annotations

import json
import hashlib
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType

import pandas as pd
import numpy as np
import pytest
import httpx
from streamlit.testing.v1 import AppTest

import cineprofile.recommender as recommender_module
import cineprofile.semantic as semantic_module
import cineprofile.hybrid_model as hybrid_module
import cineprofile.audit as audit_module
from cineprofile.candidate_pool import (
    SOURCE_BACK_CATALOG,
    SOURCE_FAVORITES,
    SOURCE_POPULARITY,
    SOURCE_SEMANTIC,
    balanced_candidate_order,
    personalize_candidate_order,
)
from cineprofile.audit import run_backtest_audit
from cineprofile.compat import (
    CineProfileVersionMismatch,
    ensure_recommendation_protocol,
    unpack_recommendation_run,
)
from cineprofile.db import connect, initialize, transaction
from cineprofile.diagnostics import diagnostic_with_ui_view
from cineprofile.imdb_import import import_ratings, normalize_ratings
from cineprofile.media_types import is_series_type
from cineprofile.preferences import (
    clear_preferences,
    load_feedback,
    load_preferences,
    save_feedback,
    save_preferences,
)
from cineprofile.personal_model import (
    ensure_personal_model,
    predict_personal_candidate,
)
from cineprofile.profile import (
    MODEL_VERSION,
    build_profile,
    profile_needs_refresh,
    render_report,
)
from cineprofile.public_rating import best_public_rating, public_rating
from cineprofile.recommender import (
    _enrich_cached,
    _passes_date_filter,
    _vote_threshold,
    recommend_movies,
    score_candidates,
)
from cineprofile.semantic import semantic_evidence
from cineprofile.result_filters import (
    filter_recommendations,
    runtime_matches,
    sort_recommendations,
)
from cineprofile.ranking import (
    build_recommendation_lists,
    rank_safe_recommendations,
    rerank_recommendations,
)
from cineprofile.settings import (
    forget_tmdb_token_file,
    read_connection_settings,
    read_tmdb_token_file,
    save_connection_settings,
    save_tmdb_token_file,
)
from cineprofile.tmdb import (
    TmdbClient,
    TmdbError,
    _store_details,
    enrich_candidates,
    enrich_library,
)
from cineprofile.watch_interest import score_watch_interest


ROOT = Path(__file__).parents[1]
SAMPLE = ROOT / "data" / "sample_ratings.csv"


def test_normalize_imdb_export() -> None:
    raw = pd.read_csv(SAMPLE, dtype=str)
    normalized = normalize_ratings(raw)
    assert len(normalized) == 6
    assert normalized.iloc[0]["imdb_id"] == "tt0133093"
    assert normalized.iloc[0]["user_rating"] == 10.0
    assert normalized.iloc[0]["genres_csv"] == "Action, Sci-Fi"


def test_import_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "cineprofile.db"
    first = import_ratings(SAMPLE, database)
    second = import_ratings(SAMPLE, database)
    assert first.imported_rows == 6
    assert first.updated_rows == 0
    assert first.unchanged_rows == 0
    assert second.imported_rows == 0
    assert second.updated_rows == 0
    assert second.unchanged_rows == 6


def test_incremental_import_reports_changes_and_preserves_local_data(
    tmp_path: Path,
) -> None:
    database = tmp_path / "incremental.db"
    import_ratings(SAMPLE, database)
    with transaction(database) as connection:
        connection.execute(
            """
            UPDATE titles
            SET metadata_status='done', tmdb_id=603, overview='Enrichi'
            WHERE imdb_id='tt0133093'
            """
        )
        connection.execute(
            """
            INSERT INTO recommendation_feedback(
              tmdb_id, title, action, updated_at
            ) VALUES (603, 'The Matrix', 'already_seen', '2026-08-13')
            """
        )
        connection.execute(
            """
            INSERT INTO profile_preferences(
              entity_type, entity_id, entity_name, adjustment, updated_at
            ) VALUES ('genres', '878', 'Science-Fiction', 1, '2026-08-13')
            """
        )

    changed_payload = SAMPLE.read_text(encoding="utf-8").replace(
        "tt0133093,10,",
        "tt0133093,9,",
        1,
    ).encode("utf-8")
    result = import_ratings(changed_payload, database)

    assert result.imported_rows == 0
    assert result.updated_rows == 1
    assert result.unchanged_rows == 5
    assert result.enriched_rows_preserved == 1
    assert result.feedback_rows_preserved == 1
    assert result.preference_rows_preserved == 1
    with connect(database) as connection:
        row = connection.execute(
            """
            SELECT user_rating, metadata_status, tmdb_id, overview
            FROM titles WHERE imdb_id='tt0133093'
            """
        ).fetchone()
    assert tuple(row) == (9.0, "done", 603, "Enrichi")


def test_database_context_manager_closes_connection(tmp_path: Path) -> None:
    database = tmp_path / "closed.db"
    initialize(database)
    with connect(database) as connection:
        connection.execute("SELECT 1").fetchone()

    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_transaction_rolls_back_and_releases_database(tmp_path: Path) -> None:
    database = tmp_path / "rollback.db"
    initialize(database)

    with pytest.raises(RuntimeError, match="annulation"):
        with transaction(database) as connection:
            connection.execute(
                """
                INSERT INTO titles(imdb_id, title, user_rating)
                VALUES ('tt0000001', 'Temporaire', 8)
                """
            )
            raise RuntimeError("annulation")

    with connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM titles"
        ).fetchone()[0] == 0
    database.unlink()
    assert not database.exists()


def test_import_deduplicates_ids_and_rejects_invalid_ratings() -> None:
    raw = pd.DataFrame(
        [
            {"Const": "tt0000001", "Title": "Ancien", "Your Rating": "7"},
            {
                "Const": "tt0000001",
                "Title": "Récent",
                "Your Rating": "8",
                "Num Votes": "1,234,567",
            },
            {"Const": "tt0000002", "Title": "Trop haut", "Your Rating": "11"},
            {"Const": "tt0000003", "Title": "Zéro", "Your Rating": "0"},
        ]
    )

    normalized = normalize_ratings(raw)

    assert normalized[["imdb_id", "title", "user_rating"]].to_dict(
        orient="records"
    ) == [
        {
            "imdb_id": "tt0000001",
            "title": "Récent",
            "user_rating": 8.0,
        }
    ]
    assert normalized.iloc[0]["num_votes"] == 1_234_567


def test_import_accepts_legacy_windows_encoding(tmp_path: Path) -> None:
    payload = (
        "Const,Title,Your Rating\n"
        "tt0000001,Le fabuleux été,8\n"
    ).encode("cp1252")

    result = import_ratings(payload, tmp_path / "cp1252.db")

    assert result.imported_rows == 1


def test_best_public_rating_prefers_the_better_supported_source() -> None:
    evidence = best_public_rating(
        tmdb_rating=9.2,
        tmdb_votes=20,
        imdb_rating=7.6,
        imdb_votes=120_000,
    )

    assert evidence.source == "imdb"
    assert evidence.adjusted_rating is not None
    assert 7.5 < evidence.adjusted_rating < 7.7


def test_profile_and_exports_without_tmdb(tmp_path: Path) -> None:
    database = tmp_path / "cineprofile.db"
    import_ratings(SAMPLE, database)
    profile = build_profile(database)
    assert profile["summary"]["rated_titles"] == 6
    assert profile["summary"]["enriched_titles"] == 0
    assert profile["dimensions"]["genres"]
    assert profile["dimensions"]["directors"]

    report = render_report(profile)
    assert "Ton empreinte cinématographique" in report
    assert "Christopher Nolan" in report

    assert profile["schema_version"] == "1.0"


def test_streamlit_app_starts(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "app.db"
    monkeypatch.setenv("CINEPROFILE_DB", str(database))
    import_ratings(SAMPLE, database)
    build_profile(database)
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.run(timeout=15)
    assert not app.exception
    assert app.title[0].value == "CineProfile"
    assert any(field.label == "Clé TMDB" for field in app.text_input)
    assert any(button.label == "Enregistrer" for button in app.button)


def test_streamlit_exposes_all_tabs_and_excludable_genres(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "app-options.db"
    monkeypatch.setenv("CINEPROFILE_DB", str(database))
    monkeypatch.setenv("TMDB_TOKEN", "fake-test-token")
    import_ratings(SAMPLE, database)
    build_profile(database)

    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)

    assert not app.exception
    assert not any(field.label == "Clé TMDB" for field in app.text_input)
    assert not any(
        button.label == "Modifier les connexions" for button in app.button
    ), "Les réglages inactifs ne doivent plus être rendus"
    assert [tab.label for tab in app.tabs] == [
        "Suggestions",
        "Ma liste",
        "Mon profil",
        "Réglages",
    ]
    excluded = next(
        widget
        for widget in app.multiselect
        if widget.label == "Genres exclus de la recherche"
    )
    assert excluded.value == ["Horreur"]
    assert {"Horreur", "Science-Fiction", "Drame", "Documentaire"} <= set(
        excluded.options
    )


def test_streamlit_renders_only_my_list_when_selected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "lazy-list.db"
    monkeypatch.setenv("CINEPROFILE_DB", str(database))
    monkeypatch.setenv("TMDB_TOKEN", "fake-test-token")
    import_ratings(SAMPLE, database)
    build_profile(database)
    save_feedback(
        {
            "tmdb_id": 991,
            "imdb_id": "tt0000991",
            "title": "Film de ma liste",
            "release_date": "2026-01-01",
            "overview": "Résumé de contrôle.",
            "genres": ["Drame"],
        },
        "watchlist",
        database,
    )

    app = AppTest.from_file(str(ROOT / "app.py"))
    app.session_state["main_navigation"] = "Ma liste"
    app.run(timeout=15)

    assert not app.exception
    assert any(
        button.label == ":material/thumb_up: 1" for button in app.button
    )
    assert not any(
        widget.label == "Genres exclus de la recherche"
        for widget in app.multiselect
    )

    next(
        button
        for button in app.button
        if button.label == ":material/thumb_down:"
    ).click()
    app.run(timeout=15)
    assert load_feedback(database)[991]["action"] == "not_interested"

    next(
        button
        for button in app.button
        if button.label == ":material/thumb_up:"
    ).click()
    app.run(timeout=15)
    assert load_feedback(database)[991]["action"] == "watchlist"


def test_tmdb_discovery_applies_real_date_filters() -> None:
    client = object.__new__(TmdbClient)
    client.region = "CH"
    captured: dict[str, object] = {}

    def fake_get(path: str, **params: object) -> dict:
        captured.update(params)
        return {
            "page": 1,
            "total_pages": 1,
            "results": [
                {"id": 1, "title": "Dans la fenêtre", "release_date": "2026-03-01"},
                {"id": 2, "title": "Trop tard", "release_date": "2028-04-01"},
                {"id": 3, "title": "Sans date", "release_date": ""},
            ],
        }

    client.get = fake_get
    results = client.discover_recent_movies(
        "2025-07-25",
        "2026-07-25",
        pages=5,
        min_votes=50,
    )

    assert [item["id"] for item in results] == [1]
    assert captured["primary_release_date.gte"] == "2025-07-25"
    assert captured["primary_release_date.lte"] == "2026-07-25"
    assert captured["vote_count.gte"] == 50
    assert "primary_release_date_gte" not in captured


def test_tmdb_retries_invalid_retry_after_then_succeeds(
    monkeypatch,
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 5:
            return httpx.Response(
                429,
                headers={"Retry-After": "valeur-invalide"},
                request=request,
            )
        return httpx.Response(200, json={"ok": True}, request=request)

    client = object.__new__(TmdbClient)
    client.language = "fr-FR"
    client.region = "CH"
    client.client = httpx.Client(
        base_url="https://api.themoviedb.org/3",
        transport=httpx.MockTransport(handler),
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        "cineprofile.tmdb.time.sleep",
        lambda delay: sleeps.append(delay),
    )
    try:
        assert client.get("/test") == {"ok": True}
    finally:
        client.close()

    assert attempts == 5
    assert sleeps == [1.0, 1.0, 1.0, 1.0]


def test_tmdb_network_failure_is_bounded_and_explicit(
    monkeypatch,
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("hors ligne", request=request)

    client = object.__new__(TmdbClient)
    client.language = "fr-FR"
    client.region = "CH"
    client.client = httpx.Client(
        base_url="https://api.themoviedb.org/3",
        transport=httpx.MockTransport(handler),
    )
    monkeypatch.setattr("cineprofile.tmdb.time.sleep", lambda _delay: None)
    try:
        with pytest.raises(TmdbError, match="plusieurs tentatives") as raised:
            client.get("/test")
    finally:
        client.close()

    assert attempts == 5
    assert raised.value.path == "/test"


def test_version_mismatch_is_reported_cleanly() -> None:
    old_module = ModuleType("old_recommender")
    old_module.RECOMMENDATION_PROTOCOL = 1

    with pytest.raises(CineProfileVersionMismatch, match="fenêtre noire"):
        ensure_recommendation_protocol(old_module)

    with pytest.raises(CineProfileVersionMismatch, match="même version"):
        unpack_recommendation_run([{"title": "ancien résultat"}])

    current_module = ModuleType("current_recommender")
    current_module.RECOMMENDATION_PROTOCOL = 17
    ensure_recommendation_protocol(current_module)
    assert unpack_recommendation_run(([{"title": "ok"}], {"returned": 1}))[1][
        "returned"
    ] == 1


def test_tmdb_token_can_be_saved_and_forgotten_locally(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    save_tmdb_token_file(env_file, "fake-local-token")
    assert read_tmdb_token_file(env_file) == "fake-local-token"
    assert "fake-local-token" in env_file.read_text(encoding="utf-8")

    forget_tmdb_token_file(env_file)
    assert read_tmdb_token_file(env_file) == ""


def test_connections_are_saved_together_without_overwriting_tmdb(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Réglage conservé\nCINEPROFILE_REGION=CH\n",
        encoding="utf-8",
    )

    save_connection_settings(
        env_file,
        tmdb_token="tmdb-secret",
        radarr_url="http://radarr.local:7878/",
        radarr_api_key="radarr-secret",
        radarr_root_folder="/movies",
        radarr_quality_profile_id="4",
    )
    settings = read_connection_settings(env_file)

    assert settings == {
        "TMDB_TOKEN": "tmdb-secret",
        "RADARR_URL": "http://radarr.local:7878",
        "RADARR_API_KEY": "radarr-secret",
        "RADARR_ROOT_FOLDER": "/movies",
        "RADARR_QUALITY_PROFILE_ID": "4",
    }
    saved = env_file.read_text(encoding="utf-8")
    assert "# Réglage conservé" in saved
    assert "CINEPROFILE_REGION=CH" in saved


def test_connection_storage_rejects_env_directory(tmp_path: Path) -> None:
    env_directory = tmp_path / ".env"
    env_directory.mkdir()

    with pytest.raises(ValueError, match="actuellement un dossier"):
        save_connection_settings(
            env_directory,
            tmdb_token="tmdb-secret",
        )


def test_affinity_index_is_stable_when_exploration_changes(tmp_path: Path) -> None:
    database = tmp_path / "stable-score.db"
    import_ratings(SAMPLE, database)
    profile = build_profile(database)
    preferred_genre = profile["dimensions"]["genres"][0]
    candidate = {
        "id": 777,
        "title": "Même film",
        "release_date": pd.Timestamp.today().date().isoformat(),
        "genres": [
            {"id": preferred_genre["id"], "name": preferred_genre["name"]}
        ],
        "keywords": {"keywords": []},
        "credits": {"cast": [], "crew": []},
        "external_ids": {"imdb_id": "tt0000777"},
        "vote_average": 8.0,
        "vote_count": 500,
    }

    cautious = score_candidates(
        profile, [candidate], database, exploration=0
    )[0]
    adventurous = score_candidates(
        profile, [candidate], database, exploration=100
    )[0]

    assert cautious["affinity_index"] == adventurous["affinity_index"]
    assert cautious["score"] == adventurous["score"]
    assert cautious["rank_score"] != adventurous["rank_score"]


def test_profile_refresh_detects_stale_counts_and_model() -> None:
    counts = {"total": 1600, "enriched": 1600, "pending": 0}
    current = {
        "model_version": MODEL_VERSION,
        "summary": {"rated_titles": 1600, "enriched_titles": 1600},
    }
    assert not profile_needs_refresh(current, counts)
    assert profile_needs_refresh(
        {**current, "model_version": "ancien-modèle"},
        counts,
    )
    assert profile_needs_refresh(
        {
            **current,
            "summary": {"rated_titles": 1600, "enriched_titles": 1500},
        },
        counts,
    )


def test_repeated_themes_are_exposed_in_profile(tmp_path: Path) -> None:
    database = tmp_path / "themes.db"
    import_ratings(SAMPLE, database)
    with transaction(database) as connection:
        connection.execute(
            "INSERT INTO keywords(tmdb_id, name) VALUES (4242, 'voyage temporel')"
        )
        connection.executemany(
            "INSERT INTO title_keywords(imdb_id, keyword_id) VALUES (?, 4242)",
            [("tt0133093",), ("tt1375666",)],
        )

    profile = build_profile(database)

    assert any(
        row["name"] == "voyage temporel"
        for row in profile["dimensions"]["keywords"]
    )


def test_single_weak_signal_cannot_claim_certain_affinity(
    tmp_path: Path,
) -> None:
    database = tmp_path / "weak-evidence.db"
    import_ratings(SAMPLE, database)
    profile = build_profile(database)
    preferred_genre = profile["dimensions"]["genres"][0]
    candidate = {
        "id": 888,
        "title": "Un seul signal",
        "genres": [
            {"id": preferred_genre["id"], "name": preferred_genre["name"]}
        ],
        "keywords": {"keywords": []},
        "credits": {"cast": [], "crew": []},
        "external_ids": {},
    }

    result = score_candidates(profile, [candidate], database)[0]

    assert 40 <= result["affinity_index"] <= 70
    assert result["confidence_label"] == "faible"


def test_recommendation_card_renders_with_explanation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "card.db"
    monkeypatch.setenv("CINEPROFILE_DB", str(database))
    monkeypatch.setenv("TMDB_TOKEN", "fake-test-token")
    import_ratings(SAMPLE, database)
    build_profile(database)
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.session_state["recommendation_ui_protocol"] = 17
    app.session_state["recommendations"] = [
        {
            "tmdb_id": 1,
            "title": "Film test",
            "release_date": "2026-01-01",
            "score": 61.0,
            "affinity_index": 61.0,
            "confidence": 45.0,
            "confidence_label": "moyenne",
            "rank_score": 64.0,
            "discovery_score": 64.0,
            "reasons": ["Affinités connues : test"],
            "poster_path": "/poster.jpg",
            "overview": "Résumé français.",
            "genres": ["Drame"],
            "original_language": "fr",
            "runtime_minutes": 100,
            "providers_ch": {},
            "directors": ["Réalisatrice test"],
            "cast": ["Acteur A", "Actrice B"],
            "bayesian_rating": 7.2,
            "vote_average": 7.4,
            "vote_count": 1200,
            "components": {
                "genres": 0.60,
                "keywords": 0.55,
                "people": 0.64,
                "semantic": 0.58,
                "explicit": None,
                "quality": 0.70,
                "freshness": 0.80,
                "novelty": 0.20,
            },
        }
    ]

    app.run(timeout=15)

    assert not app.exception
    assert any(
        metric.label == "Note publique corrigée"
        and metric.value == "7.2/10"
        for metric in app.metric
    )
    next(
        button
        for button in app.button
        if button.label == ":material/thumb_down:"
    ).click()
    app.run(timeout=15)
    assert load_feedback(database)[1]["action"] == "not_interested"


def test_horror_can_be_excluded_before_ranking(tmp_path: Path) -> None:
    database = tmp_path / "exclude-horror.db"
    profile = {
        "summary": {"average_rating": 7.0},
        "dimensions": {},
    }
    horror = {
        "id": 27,
        "title": "Film d’horreur",
        "genres": [{"id": 27, "name": "Horreur"}],
        "keywords": {"keywords": []},
        "credits": {"cast": [], "crew": []},
        "external_ids": {},
    }

    assert score_candidates(
        profile,
        [horror],
        database,
        excluded_genre_ids={27},
    ) == []


def test_any_selected_genre_can_be_excluded_before_ranking(
    tmp_path: Path,
) -> None:
    candidate = {
        "id": 878,
        "title": "Science-fiction exclue",
        "genres": [{"id": 878, "name": "Science-Fiction"}],
        "keywords": {"keywords": []},
        "credits": {"cast": [], "crew": []},
        "external_ids": {},
    }

    assert score_candidates(
        {"summary": {"average_rating": 7.0}, "dimensions": {}},
        [candidate],
        tmp_path / "exclude-sf.db",
        excluded_genre_ids={878},
    ) == []


def test_runtime_filter_treats_unknown_duration_consistently() -> None:
    assert runtime_matches(None, (30, 300))
    assert not runtime_matches(None, (80, 140))
    assert not runtime_matches("durée inconnue", (30, 300))
    assert runtime_matches(100, (80, 140))
    assert not runtime_matches(160, (80, 140))

    rows = [
        {
            "tmdb_id": 1,
            "like_probability": 80,
            "runtime_minutes": None,
            "genres": [],
            "providers_ch": {},
        },
        {
            "tmdb_id": 2,
            "like_probability": 70,
            "runtime_minutes": 100,
            "genres": [],
            "providers_ch": {},
        },
    ]
    filtered = filter_recommendations(
        rows,
        minimum_score=0,
        genres=set(),
        platforms=set(),
        languages=set(),
        runtime_range=(80, 140),
        availability="Toutes",
    )
    assert [row["tmdb_id"] for row in filtered] == [2]
    assert [
        row["tmdb_id"]
        for row in sort_recommendations(
            rows,
            field="runtime_minutes",
            descending=True,
        )
    ] == [2, 1]


def test_result_filters_combine_genre_platform_language_and_access() -> None:
    rows = [
        {
            "tmdb_id": 1,
            "like_probability": 82,
            "runtime_minutes": 105,
            "genres": ["Science-Fiction"],
            "original_language": "fr",
            "providers_ch": {"flatrate": ["Canal+"]},
        },
        {
            "tmdb_id": 2,
            "like_probability": 90,
            "runtime_minutes": 105,
            "genres": ["Drame"],
            "original_language": "en",
            "providers_ch": {"rent": ["Apple TV"]},
        },
    ]

    visible = filter_recommendations(
        rows,
        minimum_score=80,
        genres={"Science-Fiction"},
        platforms={"Canal+"},
        languages={"fr"},
        runtime_range=(90, 120),
        availability="Incluse/Gratuite",
    )

    assert [row["tmdb_id"] for row in visible] == [1]


def test_horror_exclusion_does_not_consume_analysis_budget(
    tmp_path: Path,
) -> None:
    database = tmp_path / "exclude-horror-before-details.db"
    import_ratings(SAMPLE, database)
    profile = build_profile(database)
    released = date.today().isoformat()

    class FakeClient:
        language = "fr-FR"
        region = "CH"

        def __init__(self):
            self.details_calls: list[int] = []

        def discover_recent_movies(self, *_args, **_kwargs):
            return [
                {
                    "id": 27,
                    "title": "Horreur populaire",
                    "release_date": released,
                    "genre_ids": [27],
                    "vote_count": 1000,
                    "popularity": 100,
                },
                {
                    "id": 878,
                    "title": "Science-fiction",
                    "release_date": released,
                    "genre_ids": [878],
                    "vote_count": 1000,
                    "popularity": 90,
                },
            ]

        def movie_recommendations(self, *_args, **_kwargs):
            return []

        def details(self, _media_type, tmdb_id):
            self.details_calls.append(tmdb_id)
            return {
                "id": tmdb_id,
                "title": "Science-fiction",
                "release_date": released,
                "overview": "Une exploration scientifique de l’espace.",
                "genres": [{"id": 878, "name": "Science-Fiction"}],
                "credits": {"cast": [], "crew": []},
                "keywords": {"keywords": []},
                "external_ids": {"imdb_id": "tt0000878"},
                "watch/providers": {"results": {}},
                "vote_average": 7.0,
                "vote_count": 1000,
            }

    client = FakeClient()
    results, _ = recommend_movies(
        client,
        profile,
        database,
        start_date=(date.today() - timedelta(days=365)).isoformat(),
        end_date=released,
        depth="Rapide",
        reliability="Équilibrée",
        semantic_enabled=False,
        analysis_limit=1,
        excluded_genre_ids={27},
    )

    assert client.details_calls == [878]
    assert [item["tmdb_id"] for item in results] == [878]


def test_recent_and_classic_enrichment_budgets_are_independent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "independent-lanes.db"
    import_ratings(SAMPLE, database)
    profile = build_profile(database)
    recent_release = date.today().isoformat()
    classic_release = "1940-10-15"
    candidates = [
        {
            "id": 2025,
            "title": "Film récent",
            "release_date": recent_release,
            "vote_count": 10_000,
            "popularity": 50,
            "_sources": [SOURCE_POPULARITY],
        },
        {
            "id": 1940,
            "title": "Film classique",
            "release_date": classic_release,
            "vote_count": 10_000,
            "popularity": 50,
            "_sources": [SOURCE_BACK_CATALOG],
        },
    ]
    monkeypatch.setattr(
        recommender_module,
        "_candidate_pool",
        lambda *_args, **_kwargs: (
            candidates,
            {SOURCE_POPULARITY: 1, SOURCE_BACK_CATALOG: 1},
            {
                "raw_unique_candidates": 2,
                "excluded_outside_window": 0,
                "excluded_insufficient_votes": 0,
                "excluded_genres": 0,
                "after_pre_enrichment_filters": 2,
            },
        ),
    )

    class FakeClient:
        language = "fr-FR"
        region = "CH"

        def details(self, _media_type, tmdb_id):
            is_classic = tmdb_id == 1940
            return {
                "id": tmdb_id,
                "title": "Film classique" if is_classic else "Film récent",
                "release_date": (
                    classic_release if is_classic else recent_release
                ),
                "overview": "Une histoire de test.",
                "genres": [],
                "credits": {"cast": [], "crew": []},
                "keywords": {"keywords": []},
                "external_ids": {"imdb_id": f"tt9{tmdb_id:06d}"},
                "watch/providers": {"results": {}},
                "vote_average": 8.0,
                "vote_count": 10_000,
            }

    results, diagnostics = recommend_movies(
        FakeClient(),
        profile,
        database,
        start_date=(date.today() - timedelta(days=365 * 3)).isoformat(),
        end_date=recent_release,
        depth="Rapide",
        reliability="Forte",
        semantic_enabled=False,
        analysis_limit=1,
    )
    lists = build_recommendation_lists(results)

    assert diagnostics["selected_recent_for_enrichment"] == 1
    assert diagnostics["selected_classics_for_enrichment"] == 1
    assert [item["tmdb_id"] for item in lists["safe"]] == [2025]
    assert [item["tmdb_id"] for item in lists["classics"]] == [1940]


def test_enrichment_keeps_balanced_order_before_semantic_scoring(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "balanced-before-semantic.db"
    import_ratings(SAMPLE, database)
    profile = build_profile(database)
    released = date.today().isoformat()
    candidates = [
        {
            "id": candidate_id,
            "title": f"Film {candidate_id}",
            "release_date": released,
            "vote_count": 10_000,
            "popularity": popularity,
            "_sources": [SOURCE_POPULARITY],
        }
        for candidate_id, popularity in ((7001, 100), (7002, 90))
    ]
    monkeypatch.setattr(
        recommender_module,
        "_candidate_pool",
        lambda *_args, **_kwargs: (
            candidates,
            {SOURCE_POPULARITY: 2},
            {
                "raw_unique_candidates": 2,
                "excluded_outside_window": 0,
                "excluded_insufficient_votes": 0,
                "excluded_genres": 0,
                "after_pre_enrichment_filters": 2,
            },
        ),
    )
    semantic_calls: list[list[int]] = []

    def fake_semantic_evidence(
        _database,
        items,
        _baseline,
        *,
        enabled,
    ):
        assert enabled
        semantic_calls.append([int(item["id"]) for item in items])
        return {
            int(item["id"]): {
                "score": 0.95 if int(item["id"]) == 7002 else 0.05,
                "predicted_rating": 9.5 if int(item["id"]) == 7002 else 5.0,
                "base_like_rate": 0.20,
                "confidence": 0.90,
                "similarity": 0.80,
                "positive_similarity": 0.80,
                "negative_similarity": 0.0,
                "neighbors": [],
                "engine": "semantic",
            }
            for item in items
        }

    monkeypatch.setattr(
        recommender_module,
        "semantic_evidence",
        fake_semantic_evidence,
    )

    class FakeClient:
        language = "fr-FR"
        region = "CH"

        def __init__(self) -> None:
            self.details_calls: list[int] = []

        def details(self, _media_type, tmdb_id):
            self.details_calls.append(tmdb_id)
            return {
                "id": tmdb_id,
                "title": f"Film {tmdb_id}",
                "release_date": released,
                "overview": "Une histoire test.",
                "genres": [],
                "credits": {"cast": [], "crew": []},
                "keywords": {"keywords": []},
                "external_ids": {"imdb_id": f"tt8{tmdb_id:06d}"},
                "watch/providers": {"results": {}},
                "vote_average": 8.0,
                "vote_count": 10_000,
            }

    client = FakeClient()
    results, diagnostics = recommend_movies(
        client,
        profile,
        database,
        start_date=(date.today() - timedelta(days=365)).isoformat(),
        end_date=released,
        depth="Rapide",
        reliability="Forte",
        semantic_enabled=True,
        analysis_limit=1,
    )

    assert client.details_calls == [7001]
    assert semantic_calls == [[7001]]
    assert [item["tmdb_id"] for item in results] == [7001]
    assert diagnostics["candidate_order"] == "balanced_sources_v0161"
    assert not diagnostics["semantic_retrieval_enabled"]
    assert diagnostics["semantic_final_scoring_enabled"]


def test_actor_seen_only_three_times_does_not_influence_affinity(
    tmp_path: Path,
) -> None:
    database = tmp_path / "minor-actor.db"
    profile = {
        "summary": {"average_rating": 7.0},
        "dimensions": {
            "actors": [
                {
                    "id": 123,
                    "name": "Acteur peu observé",
                    "seen": 3,
                    "average_user_rating": 9.0,
                    "confidence": 0.8,
                    "affinity": 2.0,
                }
            ]
        },
    }
    candidate = {
        "id": 1234,
        "title": "Fausse affinité humaine",
        "genres": [],
        "keywords": {"keywords": []},
        "credits": {
            "cast": [{"id": 123, "name": "Acteur peu observé"}],
            "crew": [],
        },
        "external_ids": {},
    }

    result = score_candidates(profile, [candidate], database)[0]

    assert result["affinity_index"] == 50.0
    assert result["matched_details"] == []


def test_semantic_text_includes_genres_and_themes() -> None:
    text = semantic_module._document_text(
        {
            "overview": "Une mission vers une planète inconnue.",
            "tagline": "Plus loin que les étoiles.",
            "genres": [{"id": 878, "name": "Science-Fiction"}],
            "keywords": {
                "keywords": [{"id": 42, "name": "voyage spatial"}]
            },
        }
    )

    assert "Science-Fiction" in text
    assert "voyage spatial" in text


def test_semantic_negative_neighbor_has_asymmetric_penalty() -> None:
    evidence = semantic_module._evidence_from_similarities(
        np.array([[0.55, 0.90]], dtype=float),
        [
            {"title": "Film aimé", "rating": 9.0, "residual": 2.0},
            {
                "title": "Film refusé",
                "feedback": "Pas intéressé",
                "residual": -3.0,
            },
        ],
        [{"id": 777}],
        threshold=0.30,
        engine="semantic",
    )[777]

    assert evidence["negative_similarity"] > evidence["positive_similarity"]
    assert evidence["score"] < 0.5


def test_adaptive_vote_threshold_and_date_filter() -> None:
    assert _vote_threshold(pd.Timestamp.today().date().isoformat(), "Équilibrée") == 8
    assert _vote_threshold("2010-01-01", "Équilibrée") == 250
    assert _vote_threshold("2010-01-01", "Forte") == 1200
    candidate = {"release_date": "2022-05-01"}
    assert _passes_date_filter(candidate, "2020-01-01", "2025-01-01")
    assert not _passes_date_filter(candidate, "2023-01-01", "2025-01-01")


def test_candidate_analysis_budget_favors_measured_public_sources() -> None:
    candidates = [
        {
            "id": index,
            "vote_count": 10_000 - index,
            "popularity": 100 - index,
            "_sources": [SOURCE_POPULARITY],
        }
        for index in range(1, 9)
    ]
    candidates.extend(
        [
            {
                "id": 101,
                "vote_count": 300,
                "popularity": 10,
                "_sources": [SOURCE_FAVORITES],
            },
            {
                "id": 102,
                "vote_count": 250,
                "popularity": 9,
                "_sources": [SOURCE_FAVORITES],
            },
        ]
    )

    ordered = balanced_candidate_order(candidates)

    assert [row["id"] for row in ordered[:4]] == [1, 2, 3, 4]
    assert ordered[4]["id"] == 101
    assert {row["id"] for row in ordered} >= {101, 102}


def test_discovery_diversifies_without_promoting_a_much_weaker_film() -> None:
    rows = [
        {
            "tmdb_id": 1,
            "like_probability": 90,
            "confidence": 80,
            "rank_score": 90,
            "collection_id": 10,
            "genres": ["Science-Fiction"],
            "keywords": ["espace"],
            "directors": ["A"],
            "cast": [],
        },
        {
            "tmdb_id": 2,
            "like_probability": 89,
            "confidence": 80,
            "rank_score": 89,
            "collection_id": 10,
            "genres": ["Science-Fiction"],
            "keywords": ["espace"],
            "directors": ["A"],
            "cast": [],
        },
        {
            "tmdb_id": 3,
            "like_probability": 85,
            "confidence": 75,
            "rank_score": 87,
            "collection_id": None,
            "genres": ["Mystère"],
            "keywords": ["mémoire"],
            "directors": ["B"],
            "cast": [],
        },
        {
            "tmdb_id": 4,
            "like_probability": 70,
            "confidence": 90,
            "rank_score": 95,
            "collection_id": None,
            "genres": ["Comédie"],
            "keywords": [],
            "directors": ["C"],
            "cast": [],
        },
    ]

    safe = rerank_recommendations(
        [dict(row) for row in rows],
        mode="Valeurs sûres",
    )
    discovery = rerank_recommendations(
        [dict(row) for row in rows],
        mode="Découvertes",
    )

    assert [row["tmdb_id"] for row in safe[:3]] == [1, 2, 3]
    assert discovery[0]["tmdb_id"] != 4
    assert discovery.index(
        next(row for row in discovery if row["tmdb_id"] == 4)
    ) > discovery.index(
        next(row for row in discovery if row["tmdb_id"] == 2)
    )


def test_safe_list_uses_personal_signals_as_guardrails() -> None:
    rows = [
        {
            "tmdb_id": 1,
            "title": "Très bien noté mais sans envie",
            "bayesian_rating": 8.4,
            "public_rating_reliability": 92,
            "vote_count": 10_000,
            "predicted_rating": 6.5,
            "user_baseline_rating": 6.8,
            "interest_score": 20,
            "recommendation_score": 25,
        },
        {
            "tmdb_id": 2,
            "title": "Qualité et compatibilité",
            "bayesian_rating": 7.4,
            "public_rating_reliability": 88,
            "vote_count": 8_000,
            "predicted_rating": 7.1,
            "user_baseline_rating": 6.8,
            "interest_score": 70,
            "recommendation_score": 68,
        },
        {
            "tmdb_id": 3,
            "title": "Très personnel mais note publique faible",
            "bayesian_rating": 6.1,
            "public_rating_reliability": 70,
            "vote_count": 2_000,
            "predicted_rating": 7.3,
            "user_baseline_rating": 6.8,
            "interest_score": 95,
            "recommendation_score": 90,
        },
    ]

    ranked = rank_safe_recommendations(rows)

    assert [row["tmdb_id"] for row in ranked] == [2, 1, 3]
    assert ranked[0]["safe_eligibility_label"] == "Solide pour toi"
    assert ranked[1]["safe_eligibility_label"] == "Qualité publique seulement"


def test_safe_list_falls_back_to_public_order_without_personal_evidence() -> None:
    rows = [
        {
            "tmdb_id": 1,
            "title": "Public premier",
            "bayesian_rating": 8.1,
            "public_rating_reliability": 90,
            "vote_count": 8_000,
        },
        {
            "tmdb_id": 2,
            "title": "Public second",
            "bayesian_rating": 7.8,
            "public_rating_reliability": 95,
            "vote_count": 20_000,
        },
    ]

    ranked = rank_safe_recommendations(rows)

    assert [row["tmdb_id"] for row in ranked] == [1, 2]
    assert all(
        row["safe_eligibility_label"] == "Qualité publique"
        for row in ranked
    )


def test_two_lists_share_one_pool_and_exclude_safe_top_ten() -> None:
    rows = [
        {
            "tmdb_id": index,
            "title": f"Film {index}",
            "bayesian_rating": 9.0 - index / 100,
            "public_rating_reliability": 90,
            "vote_count": 10_000 - index,
            "like_probability": 40 + index,
            "confidence": 70,
            "recommendation_score": 40 + index,
            "discovery_score": 40 + index,
            "genres": [f"Genre {index % 4}"],
            "keywords": [],
            "directors": [],
            "cast": [],
        }
        for index in range(1, 31)
    ]

    lists = build_recommendation_lists(rows)
    safe_top_ids = {row["tmdb_id"] for row in lists["safe"][:10]}
    discovery_ids = {row["tmdb_id"] for row in lists["discovery"]}

    assert len(lists["safe"]) == 30
    assert len(lists["discovery"]) == 20
    assert lists["classics"] == []
    assert safe_top_ids.isdisjoint(discovery_ids)
    assert all(row["ranking_mode"] == "Valeurs sûres" for row in lists["safe"])
    assert all(
        row["ranking_mode"] == "Découvertes pour toi"
        for row in lists["discovery"]
    )


def test_classics_never_enter_recent_recommendation_lists() -> None:
    recent = {
        "tmdb_id": 2025,
        "title": "Film récent",
        "release_date": "2025-06-01",
        "recommendation_lane": "recent",
        "bayesian_rating": 7.5,
        "public_rating_reliability": 90,
        "vote_count": 10_000,
        "recommendation_score": 70,
        "interest_score": 70,
        "discovery_score": 70,
        "like_probability": 70,
        "confidence": 70,
        "genres": [],
        "keywords": [],
        "directors": [],
        "cast": [],
    }
    classic = {
        **recent,
        "tmdb_id": 1940,
        "title": "Le classique",
        "release_date": "1940-10-15",
        "recommendation_lane": "classics",
        "bayesian_rating": 9.0,
    }

    lists = build_recommendation_lists([classic, recent])

    assert [row["tmdb_id"] for row in lists["safe"]] == [2025]
    assert all(row["tmdb_id"] != 1940 for row in lists["discovery"])
    assert [row["tmdb_id"] for row in lists["classics"]] == [1940]
    assert lists["classics"][0]["recommendation_view"] == "classics"

    # Saved 0.15 selections did not yet persist recommendation_lane.
    legacy_classic = {
        **classic,
        "recommendation_lane": None,
        "sources": ["Catalogue public plus ancien"],
    }
    legacy_lists = build_recommendation_lists([legacy_classic, recent])
    assert [row["tmdb_id"] for row in legacy_lists["safe"]] == [2025]
    assert [row["tmdb_id"] for row in legacy_lists["classics"]] == [1940]


def test_public_rating_filter_is_specific_to_safe_view() -> None:
    rows = [
        {
            "tmdb_id": 1,
            "like_probability": 90,
            "interest_score": 80,
            "bayesian_rating": 7.9,
            "genres": ["Drame"],
            "providers_ch": {},
            "original_language": "fr",
            "runtime_minutes": 100,
        },
        {
            "tmdb_id": 2,
            "like_probability": 95,
            "interest_score": 90,
            "bayesian_rating": 7.1,
            "genres": ["Drame"],
            "providers_ch": {},
            "original_language": "fr",
            "runtime_minutes": 100,
        },
    ]

    visible = filter_recommendations(
        rows,
        minimum_score=0,
        minimum_public_rating=7.5,
        genres=set(),
        platforms=set(),
        languages=set(),
        runtime_range=(30, 300),
        availability="Toutes",
    )

    assert [row["tmdb_id"] for row in visible] == [1]


def test_tmdb_rating_is_conservative_when_votes_are_sparse() -> None:
    sparse = public_rating(9.2, 39, source="tmdb")
    established = public_rating(8.5, 5_000, source="tmdb")

    assert sparse.adjusted_rating is not None
    assert sparse.adjusted_rating < 6.7
    assert sparse.reliability < 0.10
    assert established.adjusted_rating is not None
    assert established.adjusted_rating > 8.2
    assert established.reliability > 0.85


def test_manual_preference_changes_candidate_affinity(tmp_path: Path) -> None:
    database = tmp_path / "preference.db"
    import_ratings(SAMPLE, database)
    profile = build_profile(database)
    preferred_genre = profile["dimensions"]["genres"][0]
    candidate = {
        "id": 919,
        "title": "Film réglable",
        "genres": [
            {"id": preferred_genre["id"], "name": preferred_genre["name"]}
        ],
        "keywords": {"keywords": []},
        "credits": {"cast": [], "crew": []},
        "external_ids": {},
    }
    automatic = score_candidates(profile, [candidate], database)[0]
    save_preferences(
        [
            {
                "entity_type": "genres",
                "entity_id": str(preferred_genre["id"]),
                "entity_name": preferred_genre["name"],
                "adjustment": -1,
            }
        ],
        database,
    )
    reduced = score_candidates(profile, [candidate], database)[0]

    assert load_preferences(database)
    assert reduced["affinity_index"] < automatic["affinity_index"]


def test_semantic_fallback_uses_liked_and_rejected_neighbors(
    tmp_path: Path,
) -> None:
    database = tmp_path / "semantic.db"
    import_ratings(SAMPLE, database)
    with transaction(database) as connection:
        connection.executemany(
            """
            UPDATE titles SET overview=?, tagline='', metadata_status='done'
            WHERE imdb_id=?
            """,
            [
                (
                    "voyage spatial vers une planète inconnue avec un équipage",
                    "tt0133093",
                ),
                (
                    "super héros combattent une armée pour sauver le monde",
                    "tt4154796",
                ),
                (
                    "évasion de prison et longue amitié entre deux hommes",
                    "tt0111161",
                ),
            ],
        )
    profile = build_profile(database)
    candidate = {
        "id": 1001,
        "overview": "un équipage part en voyage vers une planète inconnue",
        "tagline": "",
    }
    result = semantic_evidence(
        database,
        [candidate],
        profile["summary"]["average_rating"],
        enabled=False,
    )[1001]

    assert result["engine"] == "lexical"
    assert result["neighbors"]
    assert result["neighbors"][0]["title"] == "The Matrix"
    assert result["score"] > 0.5


def test_watchlist_neighbor_is_interest_only_for_semantic_prediction() -> None:
    candidates = [{"id": 1001}]
    rated_references = [
        {
            "title": "Aimé",
            "rating": 9.0,
            "residual": 2.0,
        },
        {
            "title": "Moins aimé",
            "rating": 5.0,
            "residual": -2.0,
        },
    ]
    baseline = semantic_module._evidence_from_similarities(
        np.asarray([[0.72, 0.61]], dtype=float),
        rated_references,
        candidates,
        baseline=6.8,
        base_like_rate=0.25,
        threshold=0.30,
        engine="semantic",
    )[1001]
    with_watchlist = semantic_module._evidence_from_similarities(
        np.asarray([[0.99, 0.72, 0.61]], dtype=float),
        [
            {
                "title": "À voir proche",
                "rating": None,
                "residual": 0.0,
                "feedback": "À voir",
                "reference_role": "interest",
            },
            *rated_references,
        ],
        candidates,
        baseline=6.8,
        base_like_rate=0.25,
        threshold=0.30,
        engine="semantic",
    )[1001]

    assert with_watchlist["score"] == pytest.approx(baseline["score"])
    assert with_watchlist["predicted_rating"] == pytest.approx(
        baseline["predicted_rating"]
    )
    assert with_watchlist["neighbors"][0]["feedback"] == "À voir"


def test_multisource_search_keeps_all_scores_and_reuses_cache(
    tmp_path: Path,
) -> None:
    database = tmp_path / "multisource.db"
    import_ratings(SAMPLE, database)
    profile = build_profile(database)
    released = pd.Timestamp.today().date().isoformat()

    class FakeClient:
        language = "fr-FR"
        region = "CH"

        def __init__(self):
            self.details_calls = 0

        def discover_recent_movies(self, *_args, **_kwargs):
            return [
                {
                    "id": 501,
                    "title": "Candidat",
                    "release_date": released,
                    "vote_count": 100,
                    "popularity": 12,
                }
            ]

        def movie_recommendations(self, *_args, **_kwargs):
            return []

        def details(self, _media_type, tmdb_id):
            self.details_calls += 1
            return {
                "id": tmdb_id,
                "title": "Candidat",
                "release_date": released,
                "overview": "Une histoire entièrement nouvelle.",
                "genres": [],
                "credits": {"cast": [], "crew": []},
                "keywords": {"keywords": []},
                "external_ids": {"imdb_id": "tt0000501"},
                "watch/providers": {"results": {}},
                "vote_average": 7.0,
                "vote_count": 100,
            }

    client = FakeClient()
    first, first_diagnostics = recommend_movies(
        client,
        profile,
        database,
        start_date=(date.today() - timedelta(days=365)).isoformat(),
        end_date=released,
        depth="Rapide",
        reliability="Équilibrée",
        semantic_enabled=False,
    )
    second, second_diagnostics = recommend_movies(
        client,
        profile,
        database,
        start_date=(date.today() - timedelta(days=365)).isoformat(),
        end_date=released,
        depth="Rapide",
        reliability="Équilibrée",
        semantic_enabled=False,
    )

    assert len(first) == 1
    assert first_diagnostics["returned"] == first_diagnostics["scored"] == 1
    assert client.details_calls == 1
    assert second == first
    assert second_diagnostics["cache_hits"] == 1
    diagnostic_path = Path(str(second_diagnostics["diagnostic_path"]))
    assert diagnostic_path.is_file()
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert diagnostic["search_id"] == second_diagnostics["search_id"]
    assert diagnostic["settings"]["reliability"] == "Équilibrée"
    assert diagnostic["recommendations"][0]["tmdb_rating_raw"] == 7.0
    assert diagnostic["recommendations"][0]["public_rating_adjusted"] < 7.0
    assert "token" not in json.dumps(diagnostic, ensure_ascii=False).casefold()


def test_candidate_cache_preserves_retrieval_trace(tmp_path: Path) -> None:
    database = tmp_path / "retrieval-cache.db"
    initialize(database)

    class FakeClient:
        language = "fr-FR"
        region = "CH"

        def details(self, _media_type, tmdb_id):
            return {
                "id": tmdb_id,
                "title": "Candidat tracé",
                "genres": [],
                "keywords": {"keywords": []},
                "credits": {"cast": [], "crew": []},
                "external_ids": {},
                "watch/providers": {"results": {}},
            }

    candidate = {
        "id": 9001,
        "_sources": ["Histoires proches de tes goûts"],
        "_retrieval_score": 73.5,
        "_retrieval_confidence": 62.0,
        "_retrieval_utility": 0.718,
    }
    first, first_hits, first_downloads = _enrich_cached(
        FakeClient(),
        [candidate],
        database,
        limit=1,
    )
    second, second_hits, second_downloads = _enrich_cached(
        FakeClient(),
        [candidate],
        database,
        limit=1,
    )

    assert (first_hits, first_downloads) == (0, 1)
    assert (second_hits, second_downloads) == (1, 0)
    for result in (first[0], second[0]):
        assert result["_retrieval_score"] == 73.5
        assert result["_retrieval_confidence"] == 62.0
        assert result["_retrieval_utility"] == 0.718


def test_series_are_not_used_as_movie_recommendation_seeds(
    tmp_path: Path,
) -> None:
    database = tmp_path / "seed-types.db"
    import_ratings(SAMPLE, database)
    with transaction(database) as connection:
        connection.execute(
            """
            UPDATE titles SET tmdb_id=1399, title_type='tvSeries'
            WHERE imdb_id='tt0133093'
            """
        )
        connection.execute(
            """
            UPDATE titles SET tmdb_id=278, title_type='movie'
            WHERE imdb_id='tt0111161'
            """
        )
        connection.execute(
            """
            UPDATE titles SET tmdb_id=1400, title_type='tv'
            WHERE imdb_id='tt4154796'
            """
        )

    seeds = recommender_module._favorite_seeds(database, 10)

    assert 1399 not in seeds
    assert 1400 not in seeds
    assert 278 in seeds


def test_missing_tmdb_recommendations_do_not_abort_search(
    monkeypatch,
) -> None:
    client = object.__new__(TmdbClient)

    def missing_recommendations(*_args, **_kwargs):
        raise TmdbError(
            "TMDB 404 sur /movie/1399/recommendations",
            status_code=404,
            path="/movie/1399/recommendations",
        )

    monkeypatch.setattr(client, "get", missing_recommendations)
    assert client.movie_recommendations(1399) == []
    assert client.movie_similar(1399) == []


def test_non_404_tmdb_recommendation_error_is_not_hidden(
    monkeypatch,
) -> None:
    client = object.__new__(TmdbClient)

    def authentication_error(*_args, **_kwargs):
        raise TmdbError(
            "TMDB 401",
            status_code=401,
            path="/movie/1/recommendations",
        )

    monkeypatch.setattr(client, "get", authentication_error)
    with pytest.raises(TmdbError, match="401"):
        client.movie_recommendations(1)


def test_global_tmdb_error_does_not_poison_library_status(
    tmp_path: Path,
) -> None:
    database = tmp_path / "tmdb-auth.db"
    import_ratings(SAMPLE, database)

    class FailingClient:
        def resolve_imdb_id(self, _imdb_id):
            raise TmdbError("TMDB 401", status_code=401, path="/find")

    with pytest.raises(TmdbError, match="401"):
        enrich_library(FailingClient(), database, limit=1)

    with connect(database) as connection:
        row = connection.execute(
            """
            SELECT metadata_status, metadata_error
            FROM titles ORDER BY date_rated DESC, imdb_id LIMIT 1
            """
        ).fetchone()
    assert row["metadata_status"] == "pending"
    assert row["metadata_error"] is None


def test_candidate_enrichment_only_skips_true_404() -> None:
    class MissingClient:
        def details(self, _media_type, _tmdb_id):
            raise TmdbError("absent", status_code=404, path="/movie/1")

    class UnauthorizedClient:
        def details(self, _media_type, _tmdb_id):
            raise TmdbError("interdit", status_code=401, path="/movie/1")

    candidates = [{"id": 1, "title": "Test"}]
    assert enrich_candidates(MissingClient(), candidates) == []
    with pytest.raises(TmdbError, match="interdit"):
        enrich_candidates(UnauthorizedClient(), candidates)


def test_named_tmdb_entity_collision_keeps_valid_foreign_keys(
    tmp_path: Path,
) -> None:
    database = tmp_path / "entity-collision.db"
    import_ratings(SAMPLE, database)

    def details(tmdb_id: int, keyword_id: int) -> dict:
        return {
            "id": tmdb_id,
            "title": f"Film {tmdb_id}",
            "genres": [{"id": keyword_id, "name": "Même nom"}],
            "keywords": {
                "keywords": [{"id": keyword_id, "name": "Même thème"}]
            },
            "credits": {"cast": [], "crew": []},
            "watch/providers": {"results": {}},
            "external_ids": {},
        }

    _store_details(
        database,
        "tt0133093",
        "movie",
        details(1001, 501),
        "CH",
    )
    _store_details(
        database,
        "tt0111161",
        "movie",
        details(1002, 502),
        "CH",
    )

    with connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM genres WHERE name='Même nom'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM keywords WHERE name='Même thème'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM title_genres"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM title_keywords"
        ).fetchone()[0] == 2


def test_downloaded_diagnostic_checks_follow_visible_sort() -> None:
    recommendations = [
        {
            "tmdb_id": 1,
            "title": "Initialement premier",
            "overview_available": True,
            "tmdb_vote_count": 500,
            "like_probability": 70,
            "confidence": 80,
        },
        {
            "tmdb_id": 2,
            "title": "Affiché en premier",
            "overview_available": False,
            "tmdb_vote_count": 20,
            "like_probability": 90,
            "confidence": 20,
        },
    ]
    payload = {
        "automated_checks": [],
        "recommendations": recommendations,
    }

    updated = diagnostic_with_ui_view(
        payload,
        {
            "recommendation_view": "safe",
            "visible_tmdb_ids": [2, 1],
            "displayed_tmdb_ids": [2],
        },
        view_recommendations=[
            {
                "tmdb_id": 2,
                "recommended_rank": 1,
                "ranking_mode": "Valeurs sûres",
                "recommendation_view": "safe",
                "safe_value_score": 72.4,
                "safe_eligibility_tier": 2,
                "safe_eligibility_label": "Solide pour toi",
            },
            {
                "tmdb_id": 1,
                "recommended_rank": 2,
                "ranking_mode": "Valeurs sûres",
                "recommendation_view": "safe",
                "safe_value_score": 64.2,
                "safe_eligibility_tier": 1,
                "safe_eligibility_label": "Plausible pour toi",
            },
        ],
    )

    assert payload.get("ui_view") is None
    assert updated["automated_checks_scope"] == "résultats actuellement affichés"
    warnings = {
        row["name"]
        for row in updated["automated_checks"]
        if row["status"] == "warning"
    }
    assert warnings >= {
        "probabilités élevées avec faible confiance",
        "résumés absents dans le top 10",
        "moins de 100 votes TMDB dans le top 10",
    }
    assert all(
        row["titles"] == ["Affiché en premier"]
        for row in updated["automated_checks"]
        if row["name"] in warnings and "titles" in row
    )
    visible_first = next(
        row for row in updated["recommendations"] if row["tmdb_id"] == 2
    )
    assert visible_first["recommended_rank"] == 1
    assert visible_first["safe_eligibility_label"] == "Solide pour toi"


def test_semantic_vector_path_uses_nearest_rated_films(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "vectors.db"
    import_ratings(SAMPLE, database)
    with transaction(database) as connection:
        connection.executemany(
            """
            UPDATE titles SET overview=?, metadata_status='done'
            WHERE imdb_id=?
            """,
            [
                ("voyage spatial", "tt0133093"),
                ("amitié en prison", "tt0111161"),
                ("combat de super héros", "tt4154796"),
            ],
        )
    profile = build_profile(database)
    candidate = {"id": 77, "overview": "expédition spatiale", "tagline": ""}

    def fake_embeddings(_database, documents, **_kwargs):
        assert len(documents) == 4
        return np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.99, 0.01],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        semantic_module,
        "_load_or_create_embeddings",
        fake_embeddings,
    )
    result = semantic_evidence(
        database,
        [candidate],
        profile["summary"]["average_rating"],
        enabled=True,
    )[77]

    assert result["engine"] == "semantic"
    assert result["neighbors"][0]["title"] == "The Matrix"
    assert result["score"] > 0.5


def _seed_personal_history(database: Path, count: int = 120) -> None:
    initialize(database)
    with transaction(database) as connection:
        connection.executemany(
            "INSERT INTO genres(tmdb_id, name) VALUES (?, ?)",
            [(878, "Science-Fiction"), (18, "Drame")],
        )
        for index in range(count):
            liked = index < count // 2
            imdb_id = f"tt{9_000_000 + index}"
            connection.execute(
                """
                INSERT INTO titles(
                  imdb_id, title, title_type, year, user_rating, date_rated,
                  imdb_rating, num_votes, runtime_minutes, release_date,
                  genres_csv, overview, original_language, metadata_status,
                  enriched_at
                ) VALUES (?, ?, 'movie', ?, ?, ?, 7.0, 5000, ?, ?, ?, ?, 'en',
                          'done', ?)
                """,
                (
                    imdb_id,
                    f"Film synthétique {index}",
                    1990 + index % 35,
                    9.0 if liked else 5.0,
                    (date(2020, 1, 1) + timedelta(days=index)).isoformat(),
                    110 if liked else 145,
                    f"{1990 + index % 35}-01-01",
                    "Sci-Fi" if liked else "Drama",
                    (
                        "Une exploration spatiale cérébrale sur le temps et "
                        "la conscience."
                        if liked
                        else "Un drame biographique conventionnel et sentimental."
                    ),
                    (date(2026, 1, 1) + timedelta(days=index)).isoformat(),
                ),
            )
            connection.execute(
                "INSERT INTO title_genres(imdb_id, genre_id) VALUES (?, ?)",
                (imdb_id, 878 if liked else 18),
            )


def test_backtest_audit_is_repeated_and_read_only(
    tmp_path: Path,
) -> None:
    database = tmp_path / "audit.db"
    _seed_personal_history(database, count=160)
    with transaction(database) as connection:
        connection.execute(
            """
            INSERT INTO profile_preferences(
                entity_type, entity_id, entity_name, adjustment, updated_at
            ) VALUES ('genres', '878', 'Science-Fiction', 1, ?)
            """,
            ("2026-07-25T00:00:00Z",),
        )
        connection.execute(
            """
            INSERT INTO recommendation_feedback(
                tmdb_id, title, action, updated_at
            ) VALUES (999001, 'Film refusé', 'not_interested', ?)
            """,
            ("2026-07-25T00:00:00Z",),
        )
    with connect(database) as connection:
        before = tuple(
            connection.execute(
                """
                SELECT COUNT(*), SUM(user_rating),
                       (SELECT COUNT(*) FROM profile_preferences),
                       (SELECT COUNT(*) FROM recommendation_feedback)
                FROM titles
                """
            ).fetchone()
        )

    payload = run_backtest_audit(
        database,
        repeats=2,
        learning_curve_repeats=0,
        hybrid_variants=("structured",),
    )

    with connect(database) as connection:
        after = tuple(
            connection.execute(
                """
                SELECT COUNT(*), SUM(user_rating),
                       (SELECT COUNT(*) FROM profile_preferences),
                       (SELECT COUNT(*) FROM recommendation_feedback)
                FROM titles
                """
            ).fetchone()
        )
    assert after == before
    assert payload["integrity"]["audit_used_database_snapshot"]
    assert payload["integrity"]["source_unchanged"]
    assert len(payload["random_holdouts"]["splits"]) == 2
    assert set(payload["random_holdouts"]["engine_summaries"]) >= {
        "public_baseline",
        "legacy_v05",
        "linear_v06",
        "personal_structured_v09",
    }
    recommendation = payload["automated_findings"][
        "optimizer_recommendation"
    ]
    assert recommendation["decision"] in {"promote", "keep_default"}
    automatic = hybrid_module.active_configuration(database)
    assert automatic is not None
    assert automatic["engine"] == "personal_v09"
    assert automatic["automatic"]
    assert payload["chronological"] is not None
    reports = list((database.parent / "logs").glob("audit_backtest_*.json"))
    assert len(reports) == 1
    saved = json.loads(reports[0].read_text(encoding="utf-8"))
    assert saved["created_at"] == payload["created_at"]


def test_hybrid_model_variants_are_separate_and_reversible(
    tmp_path: Path,
) -> None:
    database = tmp_path / "hybrid-model.db"
    _seed_personal_history(database, count=160)
    items = hybrid_module.pm._load_training_items(database)
    dense = np.asarray(
        [
            [
                ((sum(ord(char) for char in str(item["title"])) + offset * 17)
                 % 101)
                / 100.0
                for offset in range(12)
            ]
            for item in items
        ],
        dtype=np.float32,
    )
    dense /= np.maximum(
        np.linalg.norm(dense, axis=1, keepdims=True),
        1e-8,
    )

    lexical = hybrid_module.fit_hybrid_model(
        items,
        None,
        variant="structured_lexical",
        fingerprint="test-lexical",
    )
    semantic = hybrid_module.fit_hybrid_model(
        items,
        dense,
        variant="structured_dense",
        fingerprint="test-semantic",
    )

    assert lexical.variant == "structured_lexical"
    assert semantic.variant == "structured_dense"
    assert lexical.selected_alpha in hybrid_module.HYBRID_ALPHAS
    assert semantic.selected_alpha in hybrid_module.HYBRID_ALPHAS
    assert lexical.metrics["alpha_search"]
    assert semantic.metrics["alpha_search"]

    hybrid_module.apply_configuration(
        database,
        variant="structured_lexical",
        audit_created_at="2026-07-26T08:00:00Z",
        selected_alpha=100.0,
    )
    active = hybrid_module.active_configuration(database)
    assert active is not None
    assert active["configuration"]["variant"] == "structured_lexical"

    hybrid_module.restore_linear_configuration(database)
    restored = hybrid_module.active_configuration(database)
    assert restored is not None
    assert restored["engine"] == "linear_v06"


def test_optimizer_requires_repeatable_non_boundary_gain() -> None:
    def summary(
        ndcg: float,
        precision: float,
        auc: float,
        mae: float,
        brier: float,
    ) -> dict:
        return {
            "ndcg_at_10": {"mean": ndcg},
            "precision_at_10": {"mean": precision},
            "ndcg_at_20": {"mean": ndcg},
            "auc": {"mean": auc},
            "mae": {"mean": mae},
            "brier": {"mean": brier},
        }

    summaries = {
        "linear_v06": summary(0.75, 0.80, 0.76, 0.65, 0.18),
        "personal_structured_lexical_v09": summary(
            0.79,
            0.81,
            0.765,
            0.64,
            0.17,
        ),
    }
    details = [
        {
            "metrics": [
                {"engine": "linear_v06", "ndcg_at_10": 0.74},
                {
                    "engine": "personal_structured_lexical_v09",
                    "ndcg_at_10": 0.79,
                },
            ],
            "hybrid": {
                "structured_lexical": {"selected_alpha": 100.0}
            },
        },
        {
            "metrics": [
                {"engine": "linear_v06", "ndcg_at_10": 0.76},
                {
                    "engine": "personal_structured_lexical_v09",
                    "ndcg_at_10": 0.80,
                },
            ],
            "hybrid": {
                "structured_lexical": {"selected_alpha": 100.0}
            },
        },
    ]
    chronological = {
        "metrics": [
            {
                "engine": "linear_v06",
                "ndcg_at_10": 0.74,
                "precision_at_10": 0.80,
            },
            {
                "engine": "personal_structured_lexical_v09",
                "ndcg_at_10": 0.75,
                "precision_at_10": 0.80,
            },
        ]
    }

    promoted = audit_module._optimizer_recommendation(
        summaries,
        details,
        chronological,
    )
    assert promoted["decision"] == "promote"
    assert promoted["selected_alpha"] == 100

    for detail in details:
        detail["hybrid"]["structured_lexical"]["selected_alpha"] = max(
            hybrid_module.HYBRID_ALPHAS
        )
    boundary = audit_module._optimizer_recommendation(
        summaries,
        details,
        chronological,
    )
    assert boundary["decision"] == "keep_default"
    assert not boundary["gates"][
        "regularization_search_has_margin"
    ]["passed"]


def test_full_hybrid_audit_uses_versioned_cache_and_all_ablations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "full-hybrid-audit.db"
    _seed_personal_history(database, count=120)

    def fake_prepare(database_path, items, *, kind="title"):
        documents = hybrid_module._embedding_documents(items, kind=kind)
        vectors = np.asarray(
            [
                [
                    (
                        sum(ord(character) for character in document["text"])
                        + dimension * 29
                    )
                    % 97
                    / 96.0
                    for dimension in range(16)
                ]
                for document in documents
            ],
            dtype=np.float32,
        )
        vectors /= np.maximum(
            np.linalg.norm(vectors, axis=1, keepdims=True),
            1e-8,
        )
        with transaction(database_path) as connection:
            connection.executemany(
                """
                INSERT INTO text_embeddings(
                  item_kind, item_id, model_name, text_hash,
                  dimensions, vector_blob, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_kind, item_id, model_name) DO UPDATE SET
                  text_hash=excluded.text_hash,
                  dimensions=excluded.dimensions,
                  vector_blob=excluded.vector_blob,
                  updated_at=excluded.updated_at
                """,
                [
                    (
                        document["kind"],
                        str(document["id"]),
                        hybrid_module.SEMANTIC_MODEL,
                        hashlib.sha256(
                            document["text"].encode("utf-8")
                        ).hexdigest(),
                        len(vector),
                        vector.tobytes(),
                        "2026-07-26T09:00:00Z",
                    )
                    for document, vector in zip(
                        documents,
                        vectors,
                        strict=True,
                    )
                ],
            )
        return vectors

    monkeypatch.setattr(
        hybrid_module,
        "prepare_dense_embeddings",
        fake_prepare,
    )

    payload = run_backtest_audit(
        database,
        repeats=2,
        learning_curve_repeats=0,
        hybrid_variants=tuple(hybrid_module.HYBRID_VARIANTS),
    )

    summaries = payload["random_holdouts"]["engine_summaries"]
    assert {
        f"personal_{variant}_v09"
        for variant in hybrid_module.HYBRID_VARIANTS
    } <= set(summaries)
    assert "semantic_retrieval_v09" in summaries
    assert payload["candidate_retrieval"]["summary"][
        "liked_recall_at_20"
    ]
    assert payload["integrity"]["source_unchanged"]
    assert payload["semantic_preparation_error"] is None
    assert payload["automated_findings"]["optimizer_recommendation"]


def test_semantic_download_failure_keeps_non_dense_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "semantic-audit-fallback.db"
    _seed_personal_history(database, count=120)

    def unavailable(*_args, **_kwargs):
        raise OSError("modèle local indisponible")

    monkeypatch.setattr(
        hybrid_module,
        "prepare_dense_embeddings",
        unavailable,
    )
    payload = run_backtest_audit(
        database,
        repeats=2,
        learning_curve_repeats=0,
        hybrid_variants=("structured", "structured_dense"),
    )

    summaries = payload["random_holdouts"]["engine_summaries"]
    assert "personal_structured_v09" in summaries
    assert "personal_structured_dense_v09" not in summaries
    assert "modèle local indisponible" in payload[
        "semantic_preparation_error"
    ]


def test_recommender_uses_only_the_explicitly_activated_challenger(
    tmp_path: Path,
) -> None:
    database = tmp_path / "active-hybrid.db"
    _seed_personal_history(database, count=160)
    profile = build_profile(database)
    hybrid_module.apply_configuration(
        database,
        variant="structured_lexical",
        audit_created_at="2026-07-26T08:00:00Z",
        selected_alpha=100.0,
    )
    candidate = {
        "id": 882001,
        "title": "Voyage intérieur",
        "release_date": "2026-01-10",
        "overview": (
            "Une exploration spatiale cérébrale sur le temps et la conscience."
        ),
        "genres": [{"id": 878, "name": "Science-Fiction"}],
        "keywords": {"keywords": []},
        "credits": {"cast": [], "crew": []},
        "external_ids": {"imdb_id": "tt8820010"},
        "vote_average": 7.4,
        "vote_count": 5000,
        "runtime": 110,
        "original_language": "en",
    }

    result = score_candidates(
        profile,
        [candidate],
        database,
        semantic_enabled=False,
    )[0]

    assert result["personal_engine"] == "personal_v09"
    assert result["personal_model_version"] == (
        recommender_module.PERSONAL_RANKER_VERSION
    )
    assert result["underlying_personal_engine"] == "personal_v09"
    assert result["personal_model_used"]
    assert hybrid_module.ensure_hybrid_model(database).model.selected_alpha == 100

    hybrid_module.restore_linear_configuration(database)
    linear_result = score_candidates(
        profile,
        [candidate],
        database,
        semantic_enabled=False,
    )[0]
    assert linear_result["personal_engine"] == "personal_v09"
    assert linear_result["underlying_personal_engine"] in {
        "linear_v06",
        "islands_v07",
    }


def test_personal_model_learns_from_existing_ratings_only(
    tmp_path: Path,
) -> None:
    database = tmp_path / "personal-model.db"
    _seed_personal_history(database)
    with connect(database) as connection:
        before = tuple(
            connection.execute(
                "SELECT COUNT(*), SUM(user_rating) FROM titles"
            ).fetchone()
        )

    state = ensure_personal_model(database)

    with connect(database) as connection:
        after = tuple(
            connection.execute(
                "SELECT COUNT(*), SUM(user_rating) FROM titles"
            ).fetchone()
        )
    assert before == after
    assert state.status == "ready"
    assert state.model is not None
    assert state.summary["rated_count"] == 120
    assert state.summary["folds"] >= 3
    assert "legacy_v05" in state.summary
    assert state.summary["new_model"]["mae"] < 1.0
    assert 0 <= state.summary["new_model"]["brier"] <= 1
    assert state.summary["new_model"]["calibration_bins"]
    assert not state.summary["challenger_validated"]
    assert state.summary["active_engine"] == "linear_v06"

    science_fiction = {
        "id": 9901,
        "title": "Nouvelle expédition",
        "release_date": "2026-06-01",
        "overview": (
            "Une exploration spatiale cérébrale sur le temps et la conscience."
        ),
        "genres": [{"id": 878, "name": "Science-Fiction"}],
        "keywords": {"keywords": []},
        "credits": {"cast": [], "crew": []},
        "vote_average": 7.0,
        "vote_count": 5000,
        "runtime": 112,
        "original_language": "en",
    }
    drama = {
        **science_fiction,
        "id": 9902,
        "title": "Biographie convenue",
        "overview": "Un drame biographique conventionnel et sentimental.",
        "genres": [{"id": 18, "name": "Drame"}],
        "runtime": 145,
    }
    liked_prediction = predict_personal_candidate(
        state.model,
        science_fiction,
    )
    disliked_prediction = predict_personal_candidate(state.model, drama)

    assert liked_prediction["predicted_rating"] > 8.0
    assert (
        liked_prediction["like_probability"]
        > disliked_prediction["like_probability"]
    )


def test_sparse_tmdb_hype_cannot_outrank_strong_personal_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "sparse-public-rating.db"
    _seed_personal_history(database)
    profile = build_profile(database)
    common = {
        "release_date": date.today().isoformat(),
        "keywords": {"keywords": []},
        "credits": {"cast": [], "crew": []},
        "external_ids": {},
        "runtime": 110,
        "original_language": "en",
    }
    sparse = {
        **common,
        "id": 9910,
        "title": "Note TMDB fragile",
        "overview": "",
        "genres": [{"id": 18, "name": "Drame"}],
        "vote_average": 9.2,
        "vote_count": 39,
    }
    established = {
        **common,
        "id": 9911,
        "title": "Correspondance solide",
        "overview": (
            "Une exploration spatiale cérébrale sur le temps et la conscience."
        ),
        "genres": [{"id": 878, "name": "Science-Fiction"}],
        "vote_average": 8.0,
        "vote_count": 5_000,
    }

    results = {
        row["title"]: row
        for row in score_candidates(
            profile,
            [sparse, established],
            database,
            semantic_enabled=False,
        )
    }
    weak = results["Note TMDB fragile"]
    strong = results["Correspondance solide"]

    assert weak["public_rating_adjusted"] < 6.7
    assert weak["public_rating_reliability"] < 10
    assert weak["confidence"] < strong["confidence"]
    assert weak["like_probability"] < strong["like_probability"]
    assert not any(
        "Décennie" in row["label"]
        for row in (
            weak["learned_positive_signals"]
            + weak["learned_negative_signals"]
            + strong["learned_positive_signals"]
            + strong["learned_negative_signals"]
        )
    )


def test_recommendations_use_calibrated_personal_prediction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "learned-recommendation.db"
    _seed_personal_history(database)
    profile = build_profile(database)
    candidate = {
        "id": 9903,
        "title": "Science-fiction test",
        "release_date": date.today().isoformat(),
        "overview": "Une exploration spatiale cérébrale sur la conscience.",
        "genres": [{"id": 878, "name": "Science-Fiction"}],
        "keywords": {"keywords": []},
        "credits": {"cast": [], "crew": []},
        "external_ids": {"imdb_id": "tt9999903"},
        "vote_average": 7.0,
        "vote_count": 5000,
        "runtime": 110,
        "original_language": "en",
    }

    result = score_candidates(
        profile,
        [candidate],
        database,
        semantic_enabled=False,
    )[0]

    assert result["personal_model_used"]
    assert result["like_probability"] == result["affinity_index"]
    assert result["predicted_rating"] > 8.0
    assert result["prediction_low"] < result["prediction_high"]

    monkeypatch.setenv("CINEPROFILE_DB", str(database))
    monkeypatch.setenv("TMDB_TOKEN", "fake-test-token")
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.session_state["recommendation_ui_protocol"] = 17
    app.session_state["recommendations"] = [result]
    app.session_state["recommendation_lists"] = {
        "safe": [],
        "discovery": [
            {
                **result,
                "ranking_mode": "Découvertes pour toi",
                "recommended_rank": 1,
                "recommendation_view": "discovery",
            }
        ],
    }
    app.session_state["recommendation_view"] = "✨ Découvertes pour toi"
    app.run(timeout=20)

    assert not app.exception
    assert any(
        metric.label == "Envie probable"
        and metric.value == result["interest_label"]
        for metric in app.metric
    )


def _interest_test_profile() -> dict:
    return {
        "summary": {
            "average_rating": 6.7,
            "high_rating_share": 0.205,
        },
        "dimensions": {"languages": []},
    }


def _interest_candidate(
    title: str,
    *,
    genres: list[str],
    keywords: list[str] | None = None,
    language: str = "en",
) -> dict:
    return {
        "title": title,
        "overview": "",
        "tagline": "",
        "genres": [
            {"id": index, "name": name}
            for index, name in enumerate(genres, start=1)
        ],
        "keywords": {
            "keywords": [
                {"id": index, "name": name}
                for index, name in enumerate(keywords or [], start=1)
            ]
        },
        "original_language": language,
    }


def _interest_score(
    candidate: dict,
    *,
    matched_details: list[dict] | None = None,
    neighbors: list[dict] | None = None,
    preferences: dict | None = None,
    probability: float = 0.30,
) -> dict:
    return score_watch_interest(
        candidate,
        profile=_interest_test_profile(),
        entities={},
        matched_details=matched_details or [],
        semantic_neighbors=neighbors or [],
        preferences=preferences or {},
        like_probability=probability,
        base_like_rate=0.205,
    )


def test_interest_layer_matches_user_acceptance_panel() -> None:
    coen = _interest_score(
        _interest_candidate(
            "Honey Don't!",
            genres=["Comédie", "Crime"],
            keywords=["dark comedy", "detective"],
        ),
        matched_details=[
            {
                "dimension": "directors",
                "name": "Ethan Coen",
                "seen": 10,
                "affinity": 0.30,
                "average_rating": 7.5,
            }
        ],
        probability=0.30,
    )
    historical = _interest_score(
        _interest_candidate(
            "Le Mage du Kremlin",
            genres=["Drame"],
            keywords=["historical fiction", "period drama"],
        ),
        probability=0.38,
    )
    fighting = _interest_score(
        _interest_candidate(
            "Road House",
            genres=["Action", "Thriller"],
            keywords=["fighting", "ufc"],
        ),
        probability=0.36,
    )
    music_biopic = _interest_score(
        _interest_candidate(
            "Un parfait inconnu",
            genres=["Drame", "Musique"],
            keywords=["biography", "singer"],
        ),
        probability=0.37,
    )
    sequel = _interest_score(
        _interest_candidate(
            "Mr Wolff 2",
            genres=["Thriller"],
            keywords=["sequel"],
        ),
        neighbors=[
            {
                "title": "Mr Wolff",
                "rating": 8,
                "similarity": 82,
            }
        ],
        probability=0.29,
    )

    assert coen["interest_score"] >= 70
    assert historical["interest_score"] < 50
    assert fighting["interest_score"] < 40
    assert music_biopic["interest_score"] < 35
    assert sequel["interest_score"] >= 55
    assert coen["recommendation_score"] > historical["recommendation_score"]
    assert coen["recommendation_score"] > fighting["recommendation_score"]
    assert historical["like_probability_lift_points"] > 0


def test_watchlist_similarity_increases_interest_without_claiming_a_rating() -> None:
    candidate = _interest_candidate(
        "Nouvelle enquête",
        genres=["Crime", "Thriller"],
        keywords=["detective"],
    )
    baseline = _interest_score(candidate, probability=0.30)
    learned = _interest_score(
        candidate,
        neighbors=[
            {
                "title": "Enquête enregistrée",
                "rating": None,
                "feedback": "À voir",
                "similarity": 75,
            }
        ],
        probability=0.30,
    )

    assert learned["interest_score"] > baseline["interest_score"]
    assert learned["recommendation_score"] > baseline["recommendation_score"]
    assert any(
        row["factor"] == "watchlist_similarity"
        for row in learned["positive_reasons"]
    )


def test_interest_preferences_can_neutralise_a_nonzero_default(
    tmp_path: Path,
) -> None:
    database = tmp_path / "interest-preferences.db"
    save_preferences(
        [
            {
                "entity_type": "interest",
                "entity_id": "biography",
                "entity_name": "Biographie et biopic",
                "adjustment": 0,
            }
        ],
        database,
    )
    preferences = load_preferences(database)
    assert preferences[("interest", "biography")]["adjustment"] == 0

    neutral = _interest_score(
        _interest_candidate(
            "Portrait",
            genres=["Drame"],
            keywords=["biography"],
        ),
        preferences=preferences,
    )
    automatic = _interest_score(
        _interest_candidate(
            "Portrait",
            genres=["Drame"],
            keywords=["biography"],
        )
    )
    assert neutral["interest_score"] > automatic["interest_score"]

    clear_preferences("interest", database)
    assert ("interest", "biography") not in load_preferences(database)


def _seed_island_history(database: Path, count_per_group: int = 40) -> None:
    initialize(database)
    groups = [
        (878, "Science-Fiction", 1001, "cérébral", 9.0),
        (878, "Science-Fiction", 1002, "sentimental", 5.0),
        (18, "Drame", 1001, "cérébral", 5.0),
        (18, "Drame", 1002, "sentimental", 9.0),
    ]
    with transaction(database) as connection:
        connection.executemany(
            "INSERT INTO genres(tmdb_id, name) VALUES (?, ?)",
            [(878, "Science-Fiction"), (18, "Drame")],
        )
        connection.executemany(
            "INSERT INTO keywords(tmdb_id, name) VALUES (?, ?)",
            [(1001, "cérébral"), (1002, "sentimental")],
        )
        index = 0
        for genre_id, genre, keyword_id, keyword, rating in groups:
            for _ in range(count_per_group):
                imdb_id = f"tt{8_000_000 + index}"
                connection.execute(
                    """
                    INSERT INTO titles(
                      imdb_id, title, title_type, year, user_rating, date_rated,
                      imdb_rating, num_votes, runtime_minutes, release_date,
                      genres_csv, overview, original_language, metadata_status,
                      enriched_at
                    ) VALUES (?, ?, 'movie', ?, ?, ?, 7.0, 5000, 110, ?,
                              ?, 'Une histoire aux choix difficiles.', 'en',
                              'done', ?)
                    """,
                    (
                        imdb_id,
                        f"Groupe {genre} {keyword} {index}",
                        1990 + index % 35,
                        rating,
                        (
                            date(2020, 1, 1) + timedelta(days=index)
                        ).isoformat(),
                        f"{1990 + index % 35}-01-01",
                        genre,
                        (
                            date(2026, 1, 1) + timedelta(days=index)
                        ).isoformat(),
                    ),
                )
                connection.execute(
                    "INSERT INTO title_genres(imdb_id, genre_id) VALUES (?, ?)",
                    (imdb_id, genre_id),
                )
                connection.execute(
                    """
                    INSERT INTO title_keywords(imdb_id, keyword_id)
                    VALUES (?, ?)
                    """,
                    (imdb_id, keyword_id),
                )
                index += 1


def test_v07_islands_win_without_mixing_scores(tmp_path: Path) -> None:
    database = tmp_path / "taste-islands.db"
    _seed_island_history(database)

    state = ensure_personal_model(database)

    assert state.status == "ready"
    assert state.model is not None
    assert state.summary["challenger_validated"]
    assert state.summary["active_engine"] == "islands_v07"
    assert state.summary["islands_v07"]["auc"] > state.summary["linear_v06"]["auc"]
    assert state.summary["islands"]["positive"]
    assert state.summary["islands"]["negative"]

    liked = {
        "id": 8801,
        "title": "Combinaison appréciée",
        "release_date": "2026-01-01",
        "overview": "Une histoire aux choix difficiles.",
        "genres": [{"id": 878, "name": "Science-Fiction"}],
        "keywords": {"keywords": [{"id": 1001, "name": "cérébral"}]},
        "credits": {"cast": [], "crew": []},
        "vote_average": 7.0,
        "vote_count": 5000,
        "runtime": 110,
        "original_language": "en",
    }
    rejected = {
        **liked,
        "id": 8802,
        "title": "Combinaison rejetée",
        "keywords": {"keywords": [{"id": 1002, "name": "sentimental"}]},
    }
    liked_prediction = predict_personal_candidate(state.model, liked)
    rejected_prediction = predict_personal_candidate(state.model, rejected)

    assert liked_prediction["engine"] == "islands_v07"
    assert liked_prediction["like_probability"] > rejected_prediction[
        "like_probability"
    ]
    assert liked_prediction["positive_island"]["representatives"]

    profile = build_profile(database)
    hybrid_module.restore_linear_configuration(database)
    scored = score_candidates(
        profile,
        [liked],
        database,
        semantic_enabled=False,
    )[0]
    assert scored["personal_engine"] == "personal_v09"
    assert scored["underlying_personal_engine"] == "islands_v07"
    assert scored["positive_island"]["representatives"]
    assert scored["negative_island"]["representatives"]
    assert scored["positive_similarity"] is not None
    assert scored["negative_similarity"] is not None


@pytest.mark.parametrize(
    "title_type",
    [
        "Série télévisée",
        "Mini-série télévisée",
        "Épisode de série télévisée",
        "tvSeries",
        "TV Mini Series",
    ],
)
def test_french_and_english_series_types_are_excluded(
    title_type: str,
) -> None:
    assert is_series_type(title_type)


@pytest.mark.parametrize(
    "title_type",
    ["Film", "Téléfilm", "Court-métrage", "Vidéo"],
)
def test_movie_like_french_types_are_kept(title_type: str) -> None:
    assert not is_series_type(title_type)


def test_semantic_retrieval_only_orders_real_source_buckets() -> None:
    candidates = [
        {
            "id": index,
            "_sources": [SOURCE_POPULARITY],
            "vote_count": 500,
            "popularity": 10,
        }
        for index in range(1, 7)
    ]
    evidence = {
        index: {
            "score": 0.15 + 0.10 * index,
            "confidence": 0.60,
        }
        for index in range(1, 7)
    }

    ordered, selected = personalize_candidate_order(
        candidates,
        evidence,
        maximum_semantic_source=3,
    )

    assert selected == 3
    assert all(
        SOURCE_SEMANTIC not in candidate["_sources"]
        for candidate in ordered
    )
    assert ordered[0]["id"] == 6


def test_public_rating_no_longer_changes_personal_affinity(
    tmp_path: Path,
) -> None:
    database = tmp_path / "public-independent-v09.db"
    _seed_personal_history(database)
    profile = build_profile(database)
    common = {
        "title": "Même histoire",
        "release_date": date.today().isoformat(),
        "overview": "Une exploration spatiale cérébrale sur la conscience.",
        "genres": [{"id": 878, "name": "Science-Fiction"}],
        "keywords": {"keywords": []},
        "credits": {"cast": [], "crew": []},
        "external_ids": {},
        "vote_count": 5000,
        "runtime": 110,
        "original_language": "en",
    }
    high_public = {**common, "id": 90101, "vote_average": 9.4}
    low_public = {**common, "id": 90102, "vote_average": 5.8}

    results = {
        row["tmdb_id"]: row
        for row in score_candidates(
            profile,
            [high_public, low_public],
            database,
            semantic_enabled=False,
        )
    }

    assert results[90101]["personal_engine"] == "personal_v09"
    assert results[90101]["public_influence_weight"] == 0.0
    assert (
        abs(
            results[90101]["like_probability"]
            - results[90102]["like_probability"]
        )
        <= 0.5
    )


def test_safe_ranking_limits_repeated_negative_genres() -> None:
    rows = [
        {
            "tmdb_id": index,
            "like_probability": probability,
            "confidence": 70,
            "rank_score": probability,
            "genres": [genre],
            "negative_genres": [genre] if genre == "Animation" else [],
            "keywords": [],
            "directors": [],
            "cast": [],
        }
        for index, probability, genre in (
            (1, 60, "Animation"),
            (2, 59, "Animation"),
            (3, 58, "Animation"),
            (4, 57, "Drame"),
            (5, 56, "Mystère"),
        )
    ]

    ranked = rerank_recommendations(rows, mode="Valeurs sûres")

    assert sum(
        "Animation" in row["genres"] for row in ranked[:4]
    ) == 2
