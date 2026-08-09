from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from cineprofile.arena import (
    ARENA_SCHEMA_VERSION,
    build_chronological_windows,
    run_offline_arena,
)
from cineprofile import hybrid_model as hm
from cineprofile.db import connect, initialize, transaction
from cineprofile.semantic import (
    cached_text_embeddings,
    embedding_cache_coverage,
)


def _items(count: int = 160) -> list[dict]:
    return [
        {
            "id": f"tt{7_000_000 + index}",
            "rating": 9.0 if index % 4 == 0 else 6.0,
            "date_rated": (date(2020, 1, 1) + timedelta(days=index)).isoformat(),
        }
        for index in range(count)
    ]


def _seed_arena_history(database: Path, count: int = 160) -> None:
    initialize(database)
    with transaction(database) as connection:
        connection.executemany(
            "INSERT INTO genres(tmdb_id, name) VALUES (?, ?)",
            [(878, "Science-Fiction"), (18, "Drame")],
        )
        for index in range(count):
            liked = index % 4 == 0
            imdb_id = f"tt{8_000_000 + index}"
            rated = (date(2020, 1, 1) + timedelta(days=index)).isoformat()
            genre_id = 878 if liked else 18
            genre_name = "Science-Fiction" if liked else "Drame"
            connection.execute(
                """
                INSERT INTO titles(
                  imdb_id, title, title_type, year, user_rating, date_rated,
                  imdb_rating, num_votes, runtime_minutes, release_date,
                  genres_csv, overview, original_language, metadata_status,
                  enriched_at
                ) VALUES (?, ?, 'movie', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'en',
                          'done', ?)
                """,
                (
                    imdb_id,
                    f"Film arène {index}",
                    1990 + index % 30,
                    9.0 if liked else 6.0,
                    rated,
                    7.2 if liked else 6.8,
                    50_000 + index,
                    105 if liked else 125,
                    f"{1990 + index % 30}-01-01",
                    genre_name,
                    (
                        "Exploration spatiale cérébrale et mystérieuse."
                        if liked
                        else "Drame sentimental biographique conventionnel."
                    ),
                    "2026-07-26T00:00:00Z",
                ),
            )
            connection.execute(
                "INSERT INTO title_genres(imdb_id, genre_id) VALUES (?, ?)",
                (imdb_id, genre_id),
            )


def test_chronological_windows_are_disjoint_and_expanding() -> None:
    items = _items()

    windows, metadata = build_chronological_windows(items)
    repeated, repeated_metadata = build_chronological_windows(items)

    assert windows == repeated
    assert metadata == repeated_metadata
    assert len(windows) == 4
    assert [len(window.train_indexes) for window in windows] == [
        80,
        100,
        120,
        140,
    ]
    assert all(len(window.test_indexes) == 20 for window in windows)
    seen_tests: set[int] = set()
    for window in windows:
        train = set(window.train_indexes)
        test = set(window.test_indexes)
        assert not train & test
        assert not seen_tests & test
        assert window.train_end < window.test_start
        seen_tests.update(test)
    assert len(seen_tests) == metadata["tested_items_once"] == 80


def test_same_day_ratings_never_cross_a_boundary() -> None:
    items = _items(180)
    shared_day = items[89]["date_rated"]
    for index in range(85, 96):
        items[index]["date_rated"] = shared_day

    windows, _ = build_chronological_windows(items, requested_windows=3)

    same_day_indexes = {
        index for index, item in enumerate(items) if item["date_rated"] == shared_day
    }
    for window in windows:
        train = set(window.train_indexes)
        test = set(window.test_indexes)
        assert not (same_day_indexes & train and same_day_indexes & test)


def test_offline_arena_is_read_only_and_saves_reproducible_report(
    tmp_path: Path,
) -> None:
    database = tmp_path / "arena.db"
    _seed_arena_history(database)
    with connect(database) as connection:
        before = tuple(
            connection.execute(
                "SELECT COUNT(*), SUM(user_rating) FROM titles"
            ).fetchone()
        )

    payload = run_offline_arena(database, requested_windows=2)

    with connect(database) as connection:
        after = tuple(
            connection.execute(
                "SELECT COUNT(*), SUM(user_rating) FROM titles"
            ).fetchone()
        )
    assert after == before
    assert payload["schema_version"] == ARENA_SCHEMA_VERSION
    assert payload["integrity"]["source_database_snapshot_used"]
    assert payload["integrity"]["source_unchanged"]
    assert len(payload["windows"]) == 2
    assert (
        payload["benchmark_status"]["catalogue_retrieval"]["status"] == "pending_step_2"
    )
    assert set(payload["engine_summaries"]) == {
        "public_rating_reference",
        "personal_structured_v09",
    }
    test_hashes = [
        window["manifest"]["test"]["manifest_hash"] for window in payload["windows"]
    ]
    assert len(test_hashes) == len(set(test_hashes))
    reports = list((tmp_path / "logs").glob("arena_baseline_*.json"))
    assert len(reports) == 1
    saved = json.loads(reports[0].read_text(encoding="utf-8"))
    assert saved["created_at"] == payload["created_at"]
    assert saved["integrity"]["source_unchanged"]


def test_dense_arena_builds_missing_vectors_only_in_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "arena.db"
    _seed_arena_history(database)
    hm.apply_configuration(
        database,
        variant="structured_dense",
        audit_created_at="2026-07-26T00:00:00Z",
        selected_alpha=50.0,
    )
    coverage = iter(
        [
            {"total": 160, "cached": 24, "missing": 136},
            {"total": 160, "cached": 160, "missing": 0},
        ]
    )
    prepared_paths: list[Path] = []
    progress_messages: list[str] = []

    def fake_coverage(
        target: str | Path | None,
        items: list[dict],
        *,
        kind: str = "title",
    ) -> dict[str, int]:
        assert kind == "title"
        assert len(items) == 160
        return next(coverage)

    def fake_prepare(
        target: str | Path | None,
        items: list[dict],
        *,
        kind: str = "title",
        cache_directory: str | Path | None = None,
        on_progress=None,
    ) -> np.ndarray:
        prepared_paths.append(Path(target or ""))
        assert Path(target or "") != database
        assert Path(cache_directory or "") == tmp_path / "models"
        assert kind == "title"
        if on_progress:
            on_progress(0, 136, "Calcul local de 136 vecteurs sémantiques")
            on_progress(136, 136, "Calcul sémantique local · 136/136")
        rows = np.asarray(
            [
                [1.0, float(index % 7), float(index % 11), float(index % 13)]
                for index in range(len(items))
            ],
            dtype=np.float32,
        )
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    monkeypatch.setattr(
        hm,
        "dense_embedding_cache_coverage",
        fake_coverage,
    )
    monkeypatch.setattr(hm, "prepare_dense_embeddings", fake_prepare)

    payload = run_offline_arena(
        database,
        requested_windows=2,
        on_progress=lambda current, total, message: progress_messages.append(
            message
        ),
    )

    assert prepared_paths
    assert payload["dense_preparation"] == {
        "required": True,
        "status": "ready",
        "total_vectors": 160,
        "cached_before": 24,
        "generated_on_snapshot": 136,
        "missing_after": 0,
        "model_cache_reused": True,
        "source_database_written": False,
        "snapshot_database_written": True,
    }
    assert payload["integrity"]["source_unchanged"]
    assert not payload["integrity"][
        "semantic_vectors_written_to_source_database"
    ]
    assert set(payload["engine_summaries"]) == {
        "public_rating_reference",
        "personal_structured_dense_v09",
    }
    assert any("Préparation sémantique" in row for row in progress_messages)
    assert any("Sémantique prête" in row for row in progress_messages)


def test_semantic_preparation_populates_an_empty_cache_with_progress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "snapshot.db"
    initialize(database)
    documents = [
        {
            "kind": "title",
            "id": f"tt{index:07d}",
            "text": f"Description de test {index}",
        }
        for index in range(1, 6)
    ]
    progress: list[tuple[int, int, str]] = []
    constructed: list[tuple[str, str]] = []

    class FakeTextEmbedding:
        def __init__(
            self,
            *,
            model_name: str,
            cache_dir: str,
            threads: int,
            providers: list[str],
        ) -> None:
            assert threads >= 1
            assert providers == ["CPUExecutionProvider"]
            constructed.append((model_name, cache_dir))

        def embed(
            self,
            texts: list[str],
            *,
            batch_size: int,
        ):
            assert batch_size == 64
            for index, _ in enumerate(texts, start=1):
                yield np.asarray(
                    [1.0, float(index), float(index % 2)],
                    dtype=np.float32,
                )

    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=FakeTextEmbedding),
    )
    model_cache = tmp_path / "shared-model"
    before = embedding_cache_coverage(database, documents)

    matrix = cached_text_embeddings(
        database,
        documents,
        cache_directory=model_cache,
        on_progress=lambda current, total, message: progress.append(
            (current, total, message)
        ),
    )
    after = embedding_cache_coverage(database, documents)
    cached_again = cached_text_embeddings(
        database,
        documents,
        cache_directory=model_cache,
    )

    assert before == {"total": 5, "cached": 0, "missing": 5}
    assert after == {"total": 5, "cached": 5, "missing": 0}
    assert matrix.shape == (5, 3)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert np.allclose(matrix, cached_again)
    assert constructed == [
        (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            str(model_cache),
        )
    ]
    assert progress[0][:2] == (0, 5)
    assert progress[-1][:2] == (5, 5)
