from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy import sparse

from cineprofile.arena import run_offline_arena
from cineprofile.db import initialize, transaction
from cineprofile.movielens import (
    MovieLensData,
    NeighborConfiguration,
    build_movielens_cache,
    canonical_imdb_id,
    ensure_movielens_dataset,
    mapped_profile,
    neighbor_predictions,
)
from cineprofile.movielens_arena import (
    MOVIELENS_ENGINE,
    _retrieval_metrics,
    run_movielens_arena,
)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _small_raw_dataset(tmp_path: Path) -> Path:
    raw = tmp_path / "ml-32m"
    raw.mkdir()
    _write_csv(
        raw / "movies.csv",
        ["movieId", "title", "genres"],
        [
            {"movieId": 1, "title": "Aimé A (2001)", "genres": "Drama"},
            {"movieId": 2, "title": "Rejeté A (2002)", "genres": "Drama"},
            {"movieId": 3, "title": "Aimé B (2003)", "genres": "Drama"},
            {"movieId": 4, "title": "Rejeté B (2004)", "genres": "Drama"},
            {"movieId": 5, "title": "Candidat (2005)", "genres": "Drama"},
        ],
    )
    _write_csv(
        raw / "links.csv",
        ["movieId", "imdbId", "tmdbId"],
        [
            {"movieId": 1, "imdbId": "0000001", "tmdbId": "1"},
            {"movieId": 2, "imdbId": "0000002", "tmdbId": "2"},
            {"movieId": 3, "imdbId": "0000003", "tmdbId": "3"},
            {"movieId": 4, "imdbId": "0000004", "tmdbId": "4"},
            {"movieId": 5, "imdbId": "0000005", "tmdbId": "5"},
        ],
    )
    rows = []
    patterns = {
        1: [5.0, 1.0, 5.0, 1.0, 5.0],
        2: [4.5, 1.5, 4.5, 1.5, 4.5],
        3: [5.0, 1.0, 4.5, 1.0, 5.0],
        4: [1.0, 5.0, 1.0, 5.0, 1.0],
        5: [1.5, 4.5, 1.5, 4.5, 1.5],
    }
    timestamp = 1_600_000_000
    for user_id, ratings in patterns.items():
        for movie_id, rating in enumerate(ratings, start=1):
            rows.append(
                {
                    "userId": user_id,
                    "movieId": movie_id,
                    "rating": rating,
                    "timestamp": timestamp,
                }
            )
            timestamp += 1
    _write_csv(
        raw / "ratings.csv",
        ["userId", "movieId", "rating", "timestamp"],
        rows,
    )
    return raw


def _load_small_cache(raw: Path) -> MovieLensData:
    cache = build_movielens_cache(raw, chunksize=10_000)
    stats = np.load(cache / "statistics.npz")
    titles = {
        int(key): value
        for key, value in json.loads(
            (cache / "movies.json").read_text(encoding="utf-8")
        ).items()
    }
    return MovieLensData(
        root=raw,
        centered_ratings=sparse.load_npz(
            cache / "centered_ratings.npz"
        ).tocsr(),
        user_means=stats["user_means"],
        movie_means=stats["movie_means"],
        movie_counts=stats["movie_counts"],
        bayesian_scores=stats["bayesian_scores"],
        movie_years=np.load(cache / "movie_years.npy"),
        movie_ids=np.load(cache / "movie_ids.npy"),
        titles=titles,
        imdb_to_movie={
            key: int(value)
            for key, value in json.loads(
                (cache / "links.json").read_text(encoding="utf-8")
            ).items()
        },
        global_mean=float(stats["global_mean"][0]),
        bayesian_prior_count=float(stats["bayesian_prior_count"][0]),
    )


def _seed_target_history(database: Path, count: int = 160) -> None:
    initialize(database)
    with transaction(database) as connection:
        for index in range(count):
            liked = index % 4 == 0
            connection.execute(
                """
                INSERT INTO titles(
                  imdb_id, title, title_type, year, user_rating, date_rated,
                  imdb_rating, num_votes, overview, original_language,
                  metadata_status, enriched_at
                ) VALUES (?, ?, 'movie', ?, ?, ?, ?, ?, ?, 'en', 'done', ?)
                """,
                (
                    f"tt{8_000_000 + index}",
                    f"Film cible {index}",
                    1990 + index % 30,
                    9.0 if liked else 6.0,
                    (
                        date(2020, 1, 1) + timedelta(days=index)
                    ).isoformat(),
                    7.2 if liked else 6.8,
                    50_000 + index,
                    "Film apprécié." if liked else "Film moyen.",
                    "2026-07-27T00:00:00Z",
                ),
            )


def _synthetic_movielens_data(root: Path) -> MovieLensData:
    users = 30
    movies = 220
    ratings = np.zeros((users + 1, movies + 1), dtype=np.float32)
    for user_id in range(1, users + 1):
        for movie_id in range(1, movies + 1):
            base = 9.0 if (movie_id - 1) % 4 == 0 else 6.0
            ratings[user_id, movie_id] = np.clip(
                base + ((user_id % 3) - 1) * 0.2,
                1.0,
                10.0,
            )
    user_means = np.mean(ratings[:, 1:], axis=1)
    centered = ratings.copy()
    centered[:, 1:] -= user_means[:, None]
    movie_means = np.mean(ratings[1:], axis=0)
    movie_counts = np.zeros(movies + 1, dtype=np.int32)
    movie_counts[1:] = users
    movie_ids = np.arange(1, movies + 1, dtype=np.int32)
    return MovieLensData(
        root=root,
        centered_ratings=sparse.csr_matrix(centered),
        user_means=user_means,
        movie_means=movie_means,
        movie_counts=movie_counts,
        bayesian_scores=movie_means,
        movie_years=np.full(movies + 1, 2000, dtype=np.int16),
        movie_ids=movie_ids,
        titles={movie_id: f"Film ML {movie_id}" for movie_id in movie_ids},
        imdb_to_movie={
            f"tt{8_000_000 + index}": index + 1 for index in range(160)
        },
        global_mean=float(np.mean(ratings[1:, 1:])),
        bayesian_prior_count=users,
    )


def test_canonical_imdb_id_preserves_modern_ids() -> None:
    assert canonical_imdb_id("tt0114709") == "tt0114709"
    assert canonical_imdb_id("114709") == "tt0114709"
    assert canonical_imdb_id("tt12345678") == "tt12345678"
    assert canonical_imdb_id("") is None


def test_dataset_download_extracts_and_verifies_official_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    files = {
        "links.csv": b"movieId,imdbId,tmdbId\n1,0000001,1\n",
        "movies.csv": b"movieId,title,genres\n1,Test (2000),Drama\n",
        "ratings.csv": (
            b"userId,movieId,rating,timestamp\n1,1,4.0,1600000000\n"
        ),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as package:
        for name, content in files.items():
            package.writestr(f"ml-32m/{name}", content)
    archive = buffer.getvalue()

    class FakeResponse:
        status_code = 200
        headers = {"content-length": str(len(archive))}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int):
            for start in range(0, len(archive), chunk_size):
                yield archive[start : start + chunk_size]

    monkeypatch.setattr(
        "cineprofile.movielens.RAW_CHECKSUMS",
        {
            name: hashlib.md5(
                content,
                usedforsecurity=False,
            ).hexdigest()
            for name, content in files.items()
        },
    )
    monkeypatch.setattr(
        "cineprofile.movielens.httpx.stream",
        lambda *args, **kwargs: FakeResponse(),
    )
    root = tmp_path / "download"

    raw = ensure_movielens_dataset(root)

    assert {
        path.name: path.read_bytes() for path in raw.glob("*.csv")
    } == files
    monkeypatch.setattr(
        "cineprofile.movielens.httpx.stream",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Le jeu vérifié ne doit pas être retéléchargé.")
        ),
    )
    assert ensure_movielens_dataset(root) == raw


def test_cache_and_neighbor_prediction_use_community_taste(
    tmp_path: Path,
) -> None:
    data = _load_small_cache(_small_raw_dataset(tmp_path))
    profile = [
        {"id": "tt0000001", "rating": 9.0},
        {"id": "tt0000002", "rating": 2.0},
        {"id": "tt0000003", "rating": 9.0},
        {"id": "tt0000004", "rating": 2.0},
    ]
    mapped, ratings = mapped_profile(data, profile)
    prediction = neighbor_predictions(
        data,
        profile,
        np.asarray([5], dtype=np.int32),
        NeighborConfiguration(
            minimum_overlap=2,
            shrinkage=1.0,
            neighbors=3,
        ),
    )

    assert mapped.tolist() == [1, 2, 3, 4]
    assert ratings.tolist() == [9.0, 2.0, 9.0, 2.0]
    assert prediction.mapped_profile_items == 4
    assert prediction.selected_neighbors == 3
    assert prediction.support_counts.tolist() == [3]
    assert prediction.predictions[0] > data.bayesian_scores[5]
    assert prediction.predictions[0] >= 8.4


def test_retrieval_metrics_do_not_label_unknown_candidates_negative() -> None:
    candidates = np.asarray([10, 20, 30, 40, 50], dtype=np.int32)
    scores = np.asarray([0.2, 0.9, 0.8, 0.1, 0.7])
    metrics = _retrieval_metrics(candidates, scores, {20, 50, 999})

    assert metrics["relevant_mapped"] == 2
    assert metrics["candidate_count"] == 5
    assert metrics["hits_at_20"] == 2
    assert metrics["recall_at_20"] == 1.0
    assert metrics["reciprocal_first_hit"] == 1.0


def test_step_2_reuses_the_exact_step_1_windows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "cineprofile.db"
    _seed_target_history(database)
    baseline = run_offline_arena(database, requested_windows=2)
    data = _synthetic_movielens_data(tmp_path / "movielens")
    monkeypatch.setattr(
        "cineprofile.movielens_arena.load_movielens_data",
        lambda root, on_progress=None: data,
    )

    payload = run_movielens_arena(database)

    assert payload["integrity"]["source_unchanged"]
    assert payload["baseline_report"]["windows_verified"] == 2
    assert len(payload["windows"]) == 2
    assert MOVIELENS_ENGINE in payload["satisfaction_engine_summaries"]
    assert [
        row["manifest"]["split_hash"] for row in payload["windows"]
    ] == [
        row["manifest"]["split_hash"] for row in baseline["windows"]
    ]
    assert payload["dataset"]["user_history_mapping_rate"] == 1.0
    reports = list((tmp_path / "logs").glob("arena_movielens_*.json"))
    assert len(reports) == 1
