from __future__ import annotations

from datetime import date

import cineprofile.candidate_pool as candidate_pool_module
from cineprofile.candidate_catalog import cached_discover, date_segments
from cineprofile.candidate_pool import (
    SOURCE_BACK_CATALOG,
    build_candidate_pool,
)
from cineprofile.db import connect, initialize
from cineprofile.tmdb import TmdbClient


def test_recent_period_is_split_by_calendar_year() -> None:
    assert date_segments("2023-08-14", "2026-08-14") == [
        ("2023-08-14", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
        ("2026-01-01", "2026-08-14"),
    ]


def test_discovery_scan_is_persisted_and_reused(tmp_path) -> None:
    database = tmp_path / "catalog.db"
    initialize(database)

    class FakeClient:
        language = "fr-FR"
        region = "CH"

        def __init__(self) -> None:
            self.calls = 0

        def discover_recent_movies(self, *_args, **_kwargs):
            self.calls += 1
            return [
                {
                    "id": 42,
                    "title": "Film conservé",
                    "release_date": "2025-02-03",
                    "vote_count": 500,
                }
            ]

    client = FakeClient()
    first = cached_discover(
        client,
        database,
        source="Test public",
        start_date="2025-01-01",
        end_date="2025-12-31",
        pages=3,
        min_votes=0,
    )
    second = cached_discover(
        client,
        database,
        source="Test public",
        start_date="2025-01-01",
        end_date="2025-12-31",
        pages=3,
        min_votes=0,
    )

    assert not first.cache_hit
    assert second.cache_hit
    assert second.items == first.items
    assert client.calls == 1
    with connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM candidate_catalog"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM candidate_catalog_scans"
        ).fetchone()[0] == 1


def test_tmdb_discovery_is_global_unless_region_is_requested() -> None:
    client = object.__new__(TmdbClient)
    client.region = "CH"
    calls: list[dict] = []

    def fake_get(_path: str, **params: object) -> dict:
        calls.append(params)
        return {"page": 1, "total_pages": 1, "results": []}

    client.get = fake_get
    today = date.today().isoformat()
    client.discover_recent_movies(today, today)
    client.discover_recent_movies(
        today,
        today,
        regional_release_dates=True,
    )

    assert "region" not in calls[0]
    assert calls[1]["region"] == "CH"


def test_back_catalogue_adds_older_candidates(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "older.db"
    initialize(database)
    monkeypatch.setattr(
        candidate_pool_module,
        "favorite_seeds",
        lambda _database, _limit: [],
    )
    monkeypatch.setattr(
        candidate_pool_module,
        "back_catalog_segments",
        lambda _start: [("1980-01-01", "1989-12-31")],
    )

    movies = [
        {
            "id": 1985,
            "title": "Pépite ancienne",
            "release_date": "1985-06-01",
            "vote_count": 5000,
        },
        {
            "id": 2025,
            "title": "Sortie récente",
            "release_date": "2025-06-01",
            "vote_count": 5000,
        },
    ]

    class FakeClient:
        language = "fr-FR"
        region = "CH"

        def discover_recent_movies(self, start, end, **_kwargs):
            return [
                movie
                for movie in movies
                if start <= movie["release_date"] <= end
            ]

    settings = {
        "discover_pages": 1,
        "quality_pages": 0,
        "catalogue_popularity_pages": 1,
        "catalogue_quality_pages": 0,
        "seed_count": 0,
        "recommendation_pages": 0,
        "similar_pages": 0,
        "creator_count": 0,
        "actor_count": 0,
        "keyword_count": 0,
        "genre_count": 0,
    }
    candidates, _, diagnostics = build_candidate_pool(
        FakeClient(),
        {"dimensions": {}},
        database,
        start_date="2023-01-01",
        end_date="2026-12-31",
        settings=settings,
        reliability="Souple",
        excluded_genre_ids=None,
        excluded_genre=lambda _candidate, _excluded: False,
        include_back_catalogue=True,
    )

    assert {candidate["id"] for candidate in candidates} == {1985, 2025}
    older = next(candidate for candidate in candidates if candidate["id"] == 1985)
    assert SOURCE_BACK_CATALOG in older["_sources"]
    assert diagnostics["back_catalogue_enabled"]
