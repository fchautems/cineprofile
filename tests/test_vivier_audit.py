from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import cineprofile.candidate_pool as candidate_pool_module
from cineprofile.candidate_pool import (
    SOURCE_POPULARITY,
    build_candidate_pool,
)
from cineprofile.db import connect, initialize
from cineprofile.imdb_import import import_ratings
from cineprofile.profile import build_profile
from cineprofile.vivier_audit import (
    evaluate_vivier_pool,
    run_vivier_audit,
    vivier_audit_report_path,
)


ROOT = Path(__file__).parents[1]
SAMPLE = ROOT / "data" / "sample_ratings.csv"


def test_vivier_metrics_measure_100_300_500_and_source_ablation() -> None:
    candidates = [
        {
            "id": candidate_id,
            "title": f"Film {candidate_id}",
            "popularity": 601 - candidate_id,
            "_sources": [SOURCE_POPULARITY],
        }
        for candidate_id in range(1, 601)
    ]
    targets = [
        {"tmdb_id": candidate_id, "rating": 9.0}
        for candidate_id in (50, 250, 450, 700)
    ]

    result = evaluate_vivier_pool(candidates, targets)

    baseline = result["baseline"]
    assert baseline["target_count"] == 4
    assert baseline["hits_at_100"] == 1
    assert baseline["hits_at_300"] == 2
    assert baseline["hits_at_500"] == 3
    assert baseline["recall_at_500"] == 0.75
    popularity_ablation = result["ablations"][SOURCE_POPULARITY]
    assert popularity_ablation["hits_at_500"] == 0
    assert result["source_target_coverage"][SOURCE_POPULARITY] == 3


def test_candidate_pool_traces_each_filter_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "trace.db"
    initialize(database)
    monkeypatch.setattr(
        candidate_pool_module,
        "favorite_seeds",
        lambda _database, _limit: [],
    )

    class FakeClient:
        def discover_recent_movies(self, *_args, **_kwargs):
            return [
                {
                    "id": 1,
                    "release_date": "2023-01-01",
                    "vote_count": 1000,
                    "genre_ids": [18],
                },
                {
                    "id": 2,
                    "release_date": "2010-01-01",
                    "vote_count": 1000,
                    "genre_ids": [18],
                },
                {
                    "id": 3,
                    "release_date": "2023-01-01",
                    "vote_count": 1,
                    "genre_ids": [18],
                },
                {
                    "id": 4,
                    "release_date": "2023-01-01",
                    "vote_count": 1000,
                    "genre_ids": [27],
                },
            ]

    settings = {
        "discover_pages": 1,
        "quality_pages": 0,
        "seed_count": 0,
        "recommendation_pages": 0,
        "similar_pages": 0,
        "creator_count": 0,
        "actor_count": 0,
        "keyword_count": 0,
        "genre_count": 0,
    }
    _, _, diagnostics = build_candidate_pool(
        FakeClient(),
        {"dimensions": {}},
        database,
        start_date="2020-01-01",
        end_date="2024-12-31",
        settings=settings,
        reliability="Forte",
        excluded_genre_ids={27},
        excluded_genre=lambda candidate, excluded: bool(
            set(candidate.get("genre_ids") or []) & set(excluded or set())
        ),
        trace_ids={1, 2, 3, 4, 5},
    )

    trace = diagnostics["candidate_trace"]
    assert trace["1"]["state"] == "eligible"
    assert trace["1"]["rank"] == 1
    assert trace["2"]["state"] == "outside_release_window"
    assert trace["3"]["state"] == "insufficient_votes"
    assert trace["4"]["state"] == "excluded_genre"
    assert trace["5"]["state"] == "absent_from_all_sources"


def test_retrieval_profile_does_not_train_or_persist(tmp_path: Path) -> None:
    database = tmp_path / "profile.db"
    import_ratings(SAMPLE, database)

    profile = build_profile(
        database,
        train_personal_model=False,
        persist=False,
    )

    assert profile["personal_model"]["status"] == "not_requested"
    assert "profile_run_id" not in profile
    with connect(database) as connection:
        profile_runs = connection.execute(
            "SELECT COUNT(*) FROM profile_runs"
        ).fetchone()[0]
        personal_models = connection.execute(
            "SELECT COUNT(*) FROM personal_models"
        ).fetchone()[0]
    assert profile_runs == 0
    assert personal_models == 0


def test_vivier_audit_uses_chronological_snapshots_and_keeps_source(
    tmp_path: Path,
) -> None:
    database = tmp_path / "history.db"
    initialize(database)
    first_day = date(2024, 1, 1)
    rows = []
    candidates = []
    for index in range(120):
        released = (first_day + timedelta(days=index)).isoformat()
        tmdb_id = 10_000 + index
        rows.append(
            (
                f"tt{index:07d}",
                f"Film {index}",
                "movie",
                2024,
                8.0,
                released,
                tmdb_id,
                released,
                "Résumé test",
                8.0,
                10_000,
                "done",
            )
        )
        candidates.append(
            {
                "id": tmdb_id,
                "title": f"Film {index}",
                "release_date": released,
                "overview": "Résumé test",
                "vote_count": 10_000,
                "vote_average": 8.0,
                "genre_ids": [18],
                "popularity": 120 - index,
            }
        )
    with connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO titles(
              imdb_id, title, title_type, year, user_rating, date_rated,
              tmdb_id, release_date, overview, tmdb_rating,
              tmdb_vote_count, metadata_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    class FakeClient:
        def discover_recent_movies(self, *_args, **_kwargs):
            return candidates

        def movie_recommendations(self, *_args, **_kwargs):
            return []

        def movie_similar(self, *_args, **_kwargs):
            return []

    payload = run_vivier_audit(
        FakeClient(),
        database,
        requested_windows=2,
        depth="Rapide",
        reliability="Souple",
    )

    assert payload["integrity"]["source_unchanged"]
    assert len(payload["windows"]) == 2
    assert payload["summary"]["measurable_liked_films"] == 40
    # Each historical snapshot removes its training films before measuring
    # candidate ranks, just like the production recommendation path.
    assert payload["summary"]["recall_at_100"] == 1.0
    assert payload["summary"]["recall_at_300"] == 1.0
    assert vivier_audit_report_path(database, payload).is_file()
