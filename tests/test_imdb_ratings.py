from __future__ import annotations

import gzip
from pathlib import Path

from cineprofile import imdb_ratings


def _snapshot(path: Path) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as output:
        output.write("tconst\taverageRating\tnumVotes\n")
        output.write("tt0000001\t7.8\t12345\n")
        output.write("tt0000002\t6.4\t987\n")
    return path


def test_ratings_for_ids_reads_only_requested_titles(tmp_path: Path) -> None:
    path = _snapshot(tmp_path / "title.ratings.tsv.gz")

    ratings = imdb_ratings.ratings_for_ids(path, {"tt0000002"})

    assert ratings == {"tt0000002": (6.4, 987)}


def test_hydrate_imdb_ratings_joins_real_rating_and_votes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = _snapshot(tmp_path / "title.ratings.tsv.gz")
    monkeypatch.setattr(imdb_ratings, "ensure_snapshot", lambda _database: path)

    rows, updated = imdb_ratings.hydrate_imdb_ratings(
        [
            {"tmdb_id": 1, "imdb_id": "tt0000001", "title": "Test"},
            {"tmdb_id": 2, "imdb_id": None, "title": "Sans IMDb"},
        ],
        tmp_path / "cineprofile.db",
    )

    assert updated == 1
    assert rows[0]["imdb_rating"] == 7.8
    assert rows[0]["imdb_vote_count"] == 12_345
    assert "imdb_rating" not in rows[1]


def test_imdb_filter_excludes_missing_and_lower_ratings() -> None:
    from cineprofile.result_filters import filter_recommendations

    rows = [
        {"tmdb_id": 1, "imdb_rating": 8.1, "genres": [], "providers_ch": {}},
        {"tmdb_id": 2, "imdb_rating": 6.9, "genres": [], "providers_ch": {}},
        {"tmdb_id": 3, "imdb_rating": None, "genres": [], "providers_ch": {}},
    ]

    visible = filter_recommendations(
        rows,
        minimum_score=0,
        minimum_imdb_rating=7.0,
        genres=set(),
        platforms=set(),
        languages=set(),
        runtime_range=(30, 300),
        availability="Toutes",
    )

    assert [row["tmdb_id"] for row in visible] == [1]
