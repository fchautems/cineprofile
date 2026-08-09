from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cineprofile.arena import run_offline_arena
from cineprofile.db import connect, initialize, transaction
from cineprofile.semantic_arena import (
    CATALOGUE_PUBLIC_ENGINE,
    _catalogue_indexes,
    _retrieval_metrics,
    run_semantic_arena,
)
from cineprofile.arena_protocol import ChronologicalWindow
from cineprofile.semantic_models import (
    SEMANTIC_MODELS,
    load_current_catalogue,
)
from cineprofile.semantic import embedding_execution_providers


def _seed_history(database: Path, count: int = 160) -> None:
    initialize(database)
    with transaction(database) as connection:
        connection.executemany(
            "INSERT INTO genres(tmdb_id, name) VALUES (?, ?)",
            [(878, "Science-Fiction"), (18, "Drame")],
        )
        for index in range(count):
            liked = index % 4 == 0
            imdb_id = f"tt{9_000_000 + index}"
            rated = (date(2020, 1, 1) + timedelta(days=index)).isoformat()
            genre_id = 878 if liked else 18
            connection.execute(
                """
                INSERT INTO titles(
                  imdb_id, title, title_type, year, user_rating, date_rated,
                  imdb_rating, num_votes, release_date, genres_csv, overview,
                  original_language, metadata_status, enriched_at, tmdb_id
                ) VALUES (?, ?, 'movie', ?, ?, ?, ?, ?, ?, ?, ?, 'fr',
                          'done', ?, ?)
                """,
                (
                    imdb_id,
                    f"Film profond {index}",
                    2000 + index % 20,
                    9.0 if liked else 6.0,
                    rated,
                    7.4 if liked else 6.6,
                    10_000 + index,
                    f"{2000 + index % 20}-01-01",
                    "Science-Fiction" if liked else "Drame",
                    (
                        "Exploration spatiale inventive et mystérieuse."
                        if liked
                        else "Drame sentimental biographique traditionnel."
                    ),
                    "2026-07-27T00:00:00Z",
                    50_000 + index,
                ),
            )
            connection.execute(
                "INSERT INTO title_genres(imdb_id, genre_id) VALUES (?, ?)",
                (imdb_id, genre_id),
            )


def _candidate_payload(tmdb_id: int) -> dict:
    return {
        "id": tmdb_id,
        "title": f"Candidat {tmdb_id}",
        "release_date": "2005-01-01",
        "vote_average": 7.1,
        "vote_count": 500,
        "overview": "Exploration spatiale contemporaine.",
        "original_language": "fr",
        "genres": [{"id": 878, "name": "Science-Fiction"}],
        "keywords": {"keywords": [{"id": 1, "name": "espace"}]},
        "credits": {
            "cast": [{"id": 10, "name": "Acteur Test"}],
            "crew": [{"id": 20, "name": "Réalisatrice Test", "job": "Director"}],
        },
        "external_ids": {},
    }


def test_current_catalogue_adds_cached_unknowns_without_duplicates(
    tmp_path: Path,
) -> None:
    database = tmp_path / "catalogue.db"
    _seed_history(database)
    with transaction(database) as connection:
        connection.execute(
            """
            INSERT INTO candidate_cache(
              tmdb_id, language, region, payload_json, fetched_at
            ) VALUES (?, 'fr-FR', 'CH', ?, ?)
            """,
            (
                999_001,
                json.dumps(_candidate_payload(999_001)),
                "2026-07-27T00:00:00Z",
            ),
        )
    from cineprofile import personal_model as pm

    rated = pm._load_training_items(database)
    catalogue, summary = load_current_catalogue(database, rated)

    assert len(catalogue) == len(rated) + 1
    assert summary["candidate_cache_movies"] == 1
    assert summary["unknown_candidates_are_not_negatives"]
    candidate = catalogue[-1]
    assert candidate.get("rating") is None
    assert candidate["entities"]["directors"] == [
        ("20", "Réalisatrice Test")
    ]


def test_retrieval_metrics_only_measure_known_future_positives() -> None:
    result = _retrieval_metrics(
        ["unknown-a", "liked", "unknown-b", "watched"],
        np.asarray([0.99, 0.90, 0.80, 0.10]),
        {"liked"},
    )

    assert result["relevant_present"] == 1
    assert result["hits_at_20"] == 1
    assert result["recall_at_20"] == 1.0


def test_catalogue_keeps_test_film_when_rating_proves_it_was_available() -> None:
    catalogue = [
        {
            "id": "train",
            "date_rated": "2024-01-01",
            "release_date": "2023-12-01",
        },
        {
            "id": "test",
            "date_rated": "2024-01-20",
            # A later release can be a national release or bad metadata.
            "release_date": "2024-06-01",
        },
        {
            "id": "future",
            "date_rated": "2024-08-01",
            "release_date": "2024-07-01",
        },
    ]
    window = ChronologicalWindow(
        window_id="chrono_test",
        train_indexes=(0,),
        test_indexes=(1,),
        train_start="2024-01-01",
        train_end="2024-01-01",
        test_start="2024-01-02",
        test_end="2024-01-31",
    )

    indexes = _catalogue_indexes(catalogue, window, rated_count=3)

    assert indexes.tolist() == [1]


def test_cuda_provider_is_explicitly_verified(monkeypatch) -> None:
    preload_directories: list[str] = []
    fake_ort = SimpleNamespace(
        preload_dlls=lambda *, directory: preload_directories.append(directory),
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    monkeypatch.setenv("CINEPROFILE_SEMANTIC_DEVICE", "cuda")
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    providers = embedding_execution_providers()

    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert preload_directories == [""]


def test_semantic_step_reuses_baseline_and_keeps_source_unchanged(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "cineprofile.db"
    _seed_history(database)
    baseline = run_offline_arena(database, requested_windows=2)
    with connect(database) as connection:
        before = tuple(
            connection.execute(
                "SELECT COUNT(*), SUM(user_rating) FROM titles"
            ).fetchone()
        )

    def fake_embeddings(
        target,
        items,
        spec,
        *,
        cache_directory=None,
        on_progress=None,
    ):
        if on_progress:
            on_progress(0, len(items), "vecteurs synthétiques")
            on_progress(len(items), len(items), "vecteurs prêts")
        offset = {
            "minilm": 1.0,
            "e5_large": 2.0,
            "bge_m3": 3.0,
        }[spec.key]
        rows = np.asarray(
            [
                [
                    1.0,
                    offset,
                    float(index % 4 == 0),
                    float(index % 7),
                    float(index % 11),
                    float(index % 13),
                ]
                for index in range(len(items))
            ],
            dtype=np.float32,
        )
        rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        return np.pad(
            rows,
            ((0, 0), (0, spec.dimensions - rows.shape[1])),
        )

    monkeypatch.setattr(
        "cineprofile.semantic_arena.prepare_model_embeddings",
        fake_embeddings,
    )
    monkeypatch.setattr(
        "cineprofile.semantic_arena.estimated_missing_download_gib",
        lambda path: 0.0,
    )

    payload = run_semantic_arena(database)

    with connect(database) as connection:
        after = tuple(
            connection.execute(
                "SELECT COUNT(*), SUM(user_rating) FROM titles"
            ).fetchone()
        )
    assert after == before
    assert payload["integrity"]["source_unchanged"]
    assert payload["baseline_report"]["windows_verified"] == 2
    assert [row["manifest"]["split_hash"] for row in payload["windows"]] == [
        row["manifest"]["split_hash"] for row in baseline["windows"]
    ]
    assert set(payload["satisfaction_engine_summaries"]) == {
        "public_rating_reference",
        "personal_structured_v09",
        *(spec.engine for spec in SEMANTIC_MODELS),
    }
    assert set(payload["catalogue_liked_summaries"]) == {
        CATALOGUE_PUBLIC_ENGINE,
        *(spec.engine for spec in SEMANTIC_MODELS),
    }
    reports = list((tmp_path / "logs").glob("arena_semantic_*.json"))
    assert len(reports) == 1
