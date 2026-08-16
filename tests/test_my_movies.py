from __future__ import annotations

from pathlib import Path

from cineprofile.preferences import (
    record_radarr_download,
    save_feedback,
    upsert_radarr_catalog_entries,
)
from cineprofile.ui_my_movies import filter_my_movies, load_my_movies


def _movie(tmdb_id: int, title: str) -> dict:
    return {
        "tmdb_id": tmdb_id,
        "imdb_id": f"tt{tmdb_id:07d}",
        "title": title,
        "release_date": "2026-01-01",
        "overview": "Résumé",
    }


def test_my_movies_unifies_feedback_and_downloads(tmp_path: Path) -> None:
    database = tmp_path / "cineprofile.db"
    watchlist = _movie(1, "À regarder")
    downloaded = _movie(2, "Envoyé")
    both = _movie(3, "Les deux")
    save_feedback(watchlist, "watchlist", database)
    record_radarr_download(downloaded, 22, database)
    save_feedback(both, "already_seen", database)
    record_radarr_download(both, 33, database, already_present=True)

    movies = load_my_movies(database)

    assert {row["tmdb_id"] for row in movies} == {1, 2, 3}
    assert [row["tmdb_id"] for row in filter_my_movies(movies, "À voir")] == [1]
    assert [row["tmdb_id"] for row in filter_my_movies(movies, "Déjà vus")] == [3]
    assert {row["tmdb_id"] for row in filter_my_movies(movies, "Radarr")} == {
        2,
        3,
    }
    assert filter_my_movies(movies, "Tous", "regarder")[0]["tmdb_id"] == 1


def test_my_movies_keeps_a_movie_discovered_directly_in_radarr(
    tmp_path: Path,
) -> None:
    database = tmp_path / "cineprofile.db"
    movie = _movie(44, "Ajouté ailleurs")

    upsert_radarr_catalog_entries(
        [movie],
        {
            44: {
                "state": "downloading",
                "detail": "Film.mkv",
                "progress": 42.0,
                "radarr_movie_id": 440,
            }
        },
        database,
    )

    loaded = load_my_movies(database)
    assert loaded[0]["tmdb_id"] == 44
    assert loaded[0]["radarr_state"] == "downloading"
    assert loaded[0]["radarr_progress"] == 42.0
