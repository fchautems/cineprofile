from __future__ import annotations

import gc
import json
import shutil
import sqlite3
import tempfile
import time
from collections import defaultdict
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import brier_score_loss, mean_absolute_error

from . import __version__
from . import hybrid_model as hm
from . import personal_model as pm
from .arena import ARENA_VERSION, _paired_comparison
from .arena_protocol import (
    DEFAULT_WINDOWS,
    ChronologicalWindow,
    build_chronological_windows,
    stable_hash,
)
from .audit import _database_fingerprint, _metric_summary
from .db import connect, initialize
from .movielens_arena import (
    _latest_baseline,
    _load_baseline,
    _reference_engine_name,
    _verify_manifests,
)
from .semantic_models import (
    SEMANTIC_MODELS,
    SemanticModelSpec,
    estimated_missing_download_gib,
    load_current_catalogue,
    prepare_model_embeddings,
)
from .semantic import embedding_execution_providers


SEMANTIC_ARENA_VERSION = "cineprofile-semantic-arena-3.0"
PUBLIC_ENGINE = "public_rating_reference"
CATALOGUE_PUBLIC_ENGINE = "current_catalogue_public_rating"
ProgressCallback = Callable[[int, int, str], None]


def _snapshot_database(source: Path, destination: Path) -> None:
    with closing(connect(source)) as incoming:
        with closing(sqlite3.connect(destination)) as outgoing:
            incoming.backup(outgoing)


def _item_available(item: dict, test_end: str) -> bool:
    boundary = date.fromisoformat(test_end[:10])
    rated_raw = str(item.get("date_rated") or "")
    try:
        if date.fromisoformat(rated_raw[:10]) <= boundary:
            # A dated user rating is stronger evidence of availability than
            # a catalogue release date, which can describe a later national
            # release, a re-release or simply contain imperfect metadata.
            return True
    except ValueError:
        pass
    raw = str(item.get("release_date") or "")
    try:
        return date.fromisoformat(raw[:10]) <= boundary
    except ValueError:
        year = item.get("year")
        try:
            return int(float(year)) <= boundary.year
        except (TypeError, ValueError, OverflowError):
            return True


def _catalogue_indexes(
    catalogue: list[dict],
    window: ChronologicalWindow,
    rated_count: int,
) -> np.ndarray:
    train_ids = {
        str(catalogue[index]["id"]) for index in window.train_indexes
    }
    result = [
        index
        for index, item in enumerate(catalogue)
        if str(item["id"]) not in train_ids
        and _item_available(item, window.test_end)
    ]
    test_ids = {
        str(catalogue[index]["id"]) for index in window.test_indexes
    }
    present = {str(catalogue[index]["id"]) for index in result}
    if not test_ids <= present:
        missing = sorted(test_ids - present)
        raise RuntimeError(
            "Le catalogue chronologique a exclu des films de la période test : "
            + ", ".join(missing[:5])
        )
    if any(index < 0 or index >= len(catalogue) for index in result):
        raise RuntimeError("Index de catalogue invalide.")
    if rated_count > len(catalogue):
        raise RuntimeError("Le catalogue ne contient pas tout l’historique.")
    return np.asarray(result, dtype=int)


def _public_catalogue_scores(
    train: list[dict],
    candidates: list[dict],
) -> np.ndarray:
    train_ratings = np.asarray(
        [float(item["rating"]) for item in train],
        dtype=float,
    )
    user_mean = float(np.mean(train_ratings))
    train_public = np.asarray(
        [pm._benchmark(item, user_mean) for item in train],
        dtype=float,
    )
    offset = float(np.mean(train_ratings - train_public))
    return np.clip(
        np.asarray(
            [pm._benchmark(item, user_mean) for item in candidates],
            dtype=float,
        )
        + offset,
        1.0,
        10.0,
    )


def _retrieval_metrics(
    keys: list[str],
    scores: np.ndarray,
    relevant_keys: set[str],
) -> dict:
    relevant = relevant_keys & set(keys)
    result: dict[str, float | int | None] = {
        "relevant_present": len(relevant),
        "candidate_count": len(keys),
    }
    if not relevant:
        for size in (20, 50, 100, 500):
            result[f"hits_at_{size}"] = 0
            result[f"recall_at_{size}"] = None
        result["reciprocal_first_hit"] = None
        result["median_relevant_rank"] = None
        result["mean_relevant_percentile"] = None
        return result

    order = np.argsort(np.asarray(scores, dtype=float))[::-1]
    ranks = {
        keys[int(index)]: rank
        for rank, index in enumerate(order, start=1)
        if keys[int(index)] in relevant
    }
    values = np.asarray(list(ranks.values()), dtype=float)
    for size in (20, 50, 100, 500):
        hits = int(np.sum(values <= size))
        result[f"hits_at_{size}"] = hits
        result[f"recall_at_{size}"] = hits / len(relevant)
    first = float(np.min(values))
    result["reciprocal_first_hit"] = 1.0 / first
    result["median_relevant_rank"] = float(np.median(values))
    result["mean_relevant_percentile"] = float(
        np.mean(1.0 - (values - 1.0) / max(1, len(keys)))
    )
    return result


def _ranking_metrics(
    predictions: list[dict],
    test: list[dict],
    *,
    engine: str,
    split: str,
) -> tuple[dict, dict]:
    scores = np.asarray(
        [float(row["like_probability"]) / 100.0 for row in predictions],
        dtype=float,
    )
    predicted = np.asarray(
        [float(row["predicted_rating"]) for row in predictions],
        dtype=float,
    )
    ratings = np.asarray([float(item["rating"]) for item in test], dtype=float)
    labels = (ratings >= pm.LIKE_THRESHOLD).astype(int)
    metrics = {
        **hm._ranking_metrics(scores, ratings),
        "mae": float(mean_absolute_error(ratings, predicted)),
        "brier": float(brier_score_loss(labels, scores)),
        "mean_probability": float(np.mean(scores)),
        "positive_rate": float(np.mean(labels)),
        "engine": engine,
        "split": split,
    }
    order = np.argsort(scores)[::-1][:20]
    details = {
        "top_20_examples": [
            {
                "title": str(test[int(index)].get("title") or ""),
                "date_rated": test[int(index)].get("date_rated"),
                "actual_rating": float(ratings[int(index)]),
                "predicted_rating": float(predicted[int(index)]),
                "score": round(float(scores[int(index)]), 4),
            }
            for index in order
        ]
    }
    return metrics, details


def _catalogue_result(
    *,
    engine: str,
    window: ChronologicalWindow,
    candidates: list[dict],
    scores: np.ndarray,
    test: list[dict],
) -> tuple[dict, dict]:
    keys = [str(item["id"]) for item in candidates]
    watched = {str(item["id"]) for item in test}
    liked = {
        str(item["id"])
        for item in test
        if float(item["rating"]) >= pm.LIKE_THRESHOLD
    }
    row = {
        "engine": engine,
        "split": window.window_id,
        "liked_future": _retrieval_metrics(keys, scores, liked),
        "watched_future": _retrieval_metrics(keys, scores, watched),
    }
    order = np.argsort(np.asarray(scores, dtype=float))[::-1][:20]
    details = {
        "candidate_count": len(candidates),
        "unknown_candidate_count": sum(
            item.get("rating") is None for item in candidates
        ),
        "top_20_examples": [
            {
                "title": str(candidates[int(index)].get("title") or ""),
                "origin": candidates[int(index)].get("catalogue_origin"),
                "score": round(float(scores[int(index)]), 4),
                "future_liked_in_this_window": (
                    str(candidates[int(index)]["id"]) in liked
                ),
                "future_watched_in_this_window": (
                    str(candidates[int(index)]["id"]) in watched
                ),
            }
            for index in order
        ],
    }
    return row, details


def _model_window(
    spec: SemanticModelSpec,
    items: list[dict],
    catalogue: list[dict],
    embeddings: np.ndarray,
    window: ChronologicalWindow,
    candidate_indexes: np.ndarray,
) -> tuple[dict, dict, dict]:
    train_indexes = np.asarray(window.train_indexes, dtype=int)
    test_indexes = np.asarray(window.test_indexes, dtype=int)
    train = [items[int(index)] for index in train_indexes]
    test = [items[int(index)] for index in test_indexes]
    model = hm.fit_hybrid_model(
        train,
        embeddings[train_indexes],
        variant="structured_dense",
        fingerprint=(
            f"{SEMANTIC_ARENA_VERSION}:{spec.key}:{window.window_id}"
        ),
    )
    test_predictions = hm.predict_hybrid_items(
        model,
        test,
        embeddings[test_indexes],
    )
    satisfaction, satisfaction_details = _ranking_metrics(
        test_predictions,
        test,
        engine=spec.engine,
        split=window.window_id,
    )
    satisfaction["selected_alpha"] = float(model.selected_alpha)
    candidates = [catalogue[int(index)] for index in candidate_indexes]
    candidate_predictions = hm.predict_hybrid_items(
        model,
        candidates,
        embeddings[candidate_indexes],
    )
    candidate_scores = np.asarray(
        [
            float(row["like_probability"]) / 100.0
            for row in candidate_predictions
        ],
        dtype=float,
    )
    retrieval, retrieval_details = _catalogue_result(
        engine=spec.engine,
        window=window,
        candidates=candidates,
        scores=candidate_scores,
        test=test,
    )
    details = {
        "selected_alpha": float(model.selected_alpha),
        "alpha_search": model.metrics["alpha_search"],
        "satisfaction": satisfaction_details,
        "catalogue": retrieval_details,
    }
    return satisfaction, retrieval, details


def _flatten_retrieval(rows: list[dict], scope: str) -> list[dict]:
    return [
        {
            "engine": row["engine"],
            "split": row["split"],
            **row[scope],
        }
        for row in rows
    ]


def _promotion_decision(
    satisfaction_rows: list[dict],
    retrieval_rows: list[dict],
    windows: list[dict],
) -> dict:
    semantic_engines = [spec.engine for spec in SEMANTIC_MODELS]
    satisfaction_summaries = {
        engine: _metric_summary(
            [row for row in satisfaction_rows if row["engine"] == engine]
        )
        for engine in semantic_engines
    }
    winner = max(
        semantic_engines,
        key=lambda engine: float(
            satisfaction_summaries[engine]
            .get("ndcg_at_20", {})
            .get("mean")
            or 0.0
        ),
    )
    comparison = _paired_comparison(
        satisfaction_rows,
        challenger=winner,
        reference=PUBLIC_ENGINE,
    )
    ndcg = comparison.get("summary", {}).get("ndcg_at_20", {})
    top_rating = comparison.get("summary", {}).get(
        "average_rating_at_20",
        {},
    )
    mae = comparison.get("summary", {}).get("mae", {})
    latest = next(
        (
            row["deltas"].get("ndcg_at_20")
            for row in comparison.get("paired_windows", [])
            if row["split"] == windows[-1]["manifest"]["window_id"]
        ),
        None,
    )
    retrieval_flat = _flatten_retrieval(retrieval_rows, "liked_future")
    retrieval_summaries = {
        engine: _metric_summary(
            [row for row in retrieval_flat if row["engine"] == engine]
        )
        for engine in [CATALOGUE_PUBLIC_ENGINE, *semantic_engines]
    }
    winner_recall = (
        retrieval_summaries[winner]
        .get("recall_at_100", {})
        .get("mean")
    )
    public_recall = (
        retrieval_summaries[CATALOGUE_PUBLIC_ENGINE]
        .get("recall_at_100", {})
        .get("mean")
    )
    recall_delta = (
        float(winner_recall) - float(public_recall)
        if winner_recall is not None and public_recall is not None
        else None
    )
    checks = [
        {
            "criterion": "NDCG@20 moyen ≥ +0,02 face à la note publique",
            "passed": float(ndcg.get("mean_delta") or -99.0) >= 0.02,
            "value": ndcg.get("mean_delta"),
        },
        {
            "criterion": "gain de NDCG@20 sur au moins 3 fenêtres sur 5",
            "passed": int(ndcg.get("wins") or 0) >= 3,
            "value": ndcg.get("wins"),
        },
        {
            "criterion": "régression récente de NDCG@20 ≥ -0,02",
            "passed": latest is not None and float(latest) >= -0.02,
            "value": latest,
        },
        {
            "criterion": "note moyenne du top 20 non dégradée",
            "passed": float(top_rating.get("mean_delta") or -99.0) >= 0.0,
            "value": top_rating.get("mean_delta"),
        },
        {
            "criterion": "MAE non dégradée de plus de 0,10",
            "passed": float(mae.get("mean_delta") or 99.0) <= 0.10,
            "value": mae.get("mean_delta"),
        },
        {
            "criterion": (
                "rappel des futurs 8+ au top 100 meilleur que la note publique"
            ),
            "passed": recall_delta is not None and recall_delta > 0.0,
            "value": recall_delta,
        },
    ]
    return {
        "best_semantic_engine": winner,
        "promote_to_reranking_test": all(row["passed"] for row in checks),
        "checks": checks,
        "winner_vs_public": comparison,
        "catalogue_liked_summaries": retrieval_summaries,
    }


def _minilm_reproduction(
    satisfaction_rows: list[dict],
    baseline: dict,
    personal_reference: str,
) -> dict:
    expected = {
        str(row["manifest"]["window_id"]): float(
            row["engines"][personal_reference]["metrics"]["ndcg_at_20"]
        )
        for row in baseline["windows"]
    }
    measured = {
        str(row["split"]): float(row["ndcg_at_20"])
        for row in satisfaction_rows
        if row["engine"] == SEMANTIC_MODELS[0].engine
    }
    deltas = {
        split: round(measured[split] - value, 9)
        for split, value in expected.items()
        if split in measured
    }
    comparable = "structured_dense" in personal_reference
    return {
        "comparable_to_step_1": comparable,
        "reference_engine": personal_reference,
        "maximum_absolute_ndcg_delta": (
            max((abs(value) for value in deltas.values()), default=None)
            if comparable
            else None
        ),
        "matches_within_1e_6": (
            bool(deltas)
            and max(abs(value) for value in deltas.values()) <= 1e-6
            if comparable
            else None
        ),
        "window_deltas": deltas if comparable else {},
    }


def _run_snapshot(
    database: Path,
    baseline: dict,
    *,
    cache_directory: Path,
    on_progress: ProgressCallback | None,
) -> dict:
    started = time.perf_counter()
    execution_providers = embedding_execution_providers()
    rated_items = pm._load_training_items(database)
    catalogue, catalogue_summary = load_current_catalogue(
        database,
        rated_items,
    )
    requested = int(
        baseline.get("dataset", {})
        .get("date_coverage", {})
        .get("requested_windows", DEFAULT_WINDOWS)
    )
    chronological, _ = build_chronological_windows(
        rated_items,
        requested_windows=requested,
    )
    manifests = _verify_manifests(rated_items, chronological, baseline)
    personal_reference = _reference_engine_name(baseline)
    satisfaction_rows: list[dict] = []
    retrieval_rows: list[dict] = []
    window_reports = [
        {
            "manifest": manifest,
            "satisfaction_ranking": {},
            "current_catalogue_retrieval_positive_unlabeled": {},
        }
        for manifest in manifests
    ]

    for position, (window, report) in enumerate(
        zip(chronological, window_reports, strict=True)
    ):
        baseline_engines = baseline["windows"][position]["engines"]
        for reference in (PUBLIC_ENGINE, personal_reference):
            metrics = baseline_engines[reference]["metrics"]
            satisfaction_rows.append(metrics)
            report["satisfaction_ranking"][reference] = {
                "metrics": metrics,
                "source": "rapport étape 1 vérifié par empreinte",
            }
        train = [rated_items[index] for index in window.train_indexes]
        test = [rated_items[index] for index in window.test_indexes]
        candidate_indexes = _catalogue_indexes(
            catalogue,
            window,
            len(rated_items),
        )
        candidates = [catalogue[int(index)] for index in candidate_indexes]
        public_scores = _public_catalogue_scores(train, candidates)
        public_retrieval, public_details = _catalogue_result(
            engine=CATALOGUE_PUBLIC_ENGINE,
            window=window,
            candidates=candidates,
            scores=public_scores,
            test=test,
        )
        retrieval_rows.append(public_retrieval)
        report["current_catalogue_retrieval_positive_unlabeled"][
            CATALOGUE_PUBLIC_ENGINE
        ] = {
            "metrics": public_retrieval,
            "details": public_details,
        }

    total_phases = len(SEMANTIC_MODELS) * (1 + len(chronological))
    phase = 0
    model_preparation: list[dict] = []
    for spec in SEMANTIC_MODELS:
        if on_progress:
            on_progress(
                phase,
                total_phases,
                f"{spec.label} · préparation locale",
            )

        def embedding_progress(
            current: int,
            total: int,
            message: str,
        ) -> None:
            if not on_progress:
                return
            percent = 100 if total <= 0 else round(100 * current / total)
            on_progress(
                phase,
                total_phases,
                f"{spec.label} · {percent}% · {message}",
            )

        model_started = time.perf_counter()
        embeddings = prepare_model_embeddings(
            database,
            catalogue,
            spec,
            cache_directory=cache_directory,
            on_progress=embedding_progress,
        )
        phase += 1
        model_preparation.append(
            {
                "key": spec.key,
                "label": spec.label,
                "model_name": spec.model_name,
                "dimensions": int(embeddings.shape[1]),
                "catalogue_vectors": int(embeddings.shape[0]),
                "download_size_gib": spec.download_gib,
                "elapsed_seconds": round(
                    time.perf_counter() - model_started,
                    3,
                ),
                "execution_providers": execution_providers,
                "vectors_written_to_snapshot_only": True,
            }
        )
        for position, (window, report) in enumerate(
            zip(chronological, window_reports, strict=True)
        ):
            if on_progress:
                on_progress(
                    phase,
                    total_phases,
                    (
                        f"{spec.label} · fenêtre {position + 1}/"
                        f"{len(chronological)} · "
                        f"{window.test_start} → {window.test_end}"
                    ),
                )
            candidate_indexes = _catalogue_indexes(
                catalogue,
                window,
                len(rated_items),
            )
            satisfaction, retrieval, details = _model_window(
                spec,
                rated_items,
                catalogue,
                embeddings,
                window,
                candidate_indexes,
            )
            satisfaction_rows.append(satisfaction)
            retrieval_rows.append(retrieval)
            report["satisfaction_ranking"][spec.engine] = {
                "metrics": satisfaction,
                "details": details["satisfaction"],
                "selection": {
                    "selected_alpha": details["selected_alpha"],
                    "alpha_search": details["alpha_search"],
                },
            }
            report["current_catalogue_retrieval_positive_unlabeled"][
                spec.engine
            ] = {
                "metrics": retrieval,
                "details": details["catalogue"],
            }
            phase += 1
        del embeddings
        gc.collect()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in satisfaction_rows:
        grouped[str(row["engine"])].append(row)
    satisfaction_summaries = {
        engine: _metric_summary(rows)
        for engine, rows in grouped.items()
    }
    retrieval_flat = _flatten_retrieval(retrieval_rows, "liked_future")
    retrieval_grouped: dict[str, list[dict]] = defaultdict(list)
    for row in retrieval_flat:
        retrieval_grouped[str(row["engine"])].append(row)
    retrieval_summaries = {
        engine: _metric_summary(rows)
        for engine, rows in retrieval_grouped.items()
    }
    if on_progress:
        on_progress(total_phases, total_phases, "Rapport sémantique terminé")
    return {
        "schema_version": 1,
        "arena_version": SEMANTIC_ARENA_VERSION,
        "baseline_arena_version": ARENA_VERSION,
        "app_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": (
            "Comparer MiniLM, Multilingual E5 Large et BGE-M3 sur les mêmes "
            "fenêtres chronologiques, sans règle de goût codée en dur."
        ),
        "methodology": {
            "windows": (
                "Les cinq fenêtres et leurs empreintes viennent de l’étape 1."
            ),
            "training": (
                "Pour chaque fenêtre, le petit modèle personnel ne voit que "
                "les notes antérieures. Son alpha est choisi dans ce passé."
            ),
            "controlled_variable": (
                "La représentation profonde du texte est la seule différence "
                "entre les trois challengers."
            ),
            "candidate_catalogue": (
                "Historique noté non encore vu à la date de la fenêtre, "
                "complété par les films enrichis du cache courant."
            ),
            "positive_unlabeled_rule": (
                "Un film non noté est inconnu, jamais considéré comme négatif."
            ),
            "metadata_time_limit": (
                "Les métadonnées sont celles disponibles aujourd’hui ; les "
                "dates de sortie empêchent seulement les films pas encore "
                "sortis d’entrer dans une fenêtre."
            ),
        },
        "baseline_report": {
            "created_at": baseline.get("created_at"),
            "windows_verified": len(manifests),
            "split_hashes": [
                row["split_hash"] for row in manifests
            ],
            "personal_reference": personal_reference,
        },
        "catalogue": catalogue_summary,
        "model_preparation": model_preparation,
        "execution": {
            "requested_device": (
                "cuda"
                if execution_providers[0] == "CUDAExecutionProvider"
                else "cpu"
            ),
            "onnx_execution_providers": execution_providers,
        },
        "windows": window_reports,
        "satisfaction_engine_summaries": satisfaction_summaries,
        "catalogue_liked_summaries": retrieval_summaries,
        "model_comparisons": {
            spec.engine: {
                reference: _paired_comparison(
                    satisfaction_rows,
                    challenger=spec.engine,
                    reference=reference,
                )
                for reference in (PUBLIC_ENGINE, personal_reference)
            }
            for spec in SEMANTIC_MODELS
        },
        "minilm_step_1_reproduction": _minilm_reproduction(
            satisfaction_rows,
            baseline,
            personal_reference,
        ),
        "promotion_decision": _promotion_decision(
            satisfaction_rows,
            retrieval_rows,
            window_reports,
        ),
        "privacy": (
            "Les textes sont encodés localement. Le rapport contient des "
            "métriques, des empreintes et au maximum vingt exemples par "
            "moteur et fenêtre, jamais l’historique intégral."
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def semantic_report_path(
    database: str | Path | None,
    payload: dict,
) -> Path:
    target = Path(database or "data/cineprofile.db").parent / "logs"
    target.mkdir(parents=True, exist_ok=True)
    try:
        created = datetime.fromisoformat(str(payload["created_at"]))
        stamp = created.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    except ValueError:
        stamp = stable_hash(str(payload.get("created_at")))[:20]
    return target / f"arena_semantic_{stamp}.json"


def save_semantic_report(
    database: str | Path | None,
    payload: dict,
) -> Path:
    path = semantic_report_path(database, payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def run_semantic_arena(
    database: str | Path | None = None,
    *,
    baseline_report: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict:
    original = Path(database or "data/cineprofile.db")
    initialize(original)
    baseline_path = (
        Path(baseline_report)
        if baseline_report is not None
        else _latest_baseline(original)
    )
    baseline = _load_baseline(baseline_path)
    model_cache = original.parent / "models"
    model_cache.mkdir(parents=True, exist_ok=True)
    missing_gib = estimated_missing_download_gib(model_cache)
    free_gib = shutil.disk_usage(model_cache).free / (1024**3)
    required_gib = missing_gib * 1.10 + 0.5
    if free_gib < required_gib:
        raise RuntimeError(
            f"Espace disque insuffisant : {free_gib:.1f} Gio libres, "
            f"environ {required_gib:.1f} Gio nécessaires pour les modèles "
            "encore absents."
        )
    fingerprint_before = _database_fingerprint(original)
    with tempfile.TemporaryDirectory(
        prefix="cineprofile-semantic-arena-",
        ignore_cleanup_errors=True,
    ) as directory:
        snapshot = Path(directory) / "semantic_arena.db"
        _snapshot_database(original, snapshot)
        payload = _run_snapshot(
            snapshot,
            baseline,
            cache_directory=model_cache,
            on_progress=on_progress,
        )
        fingerprint_after = _database_fingerprint(original)
        unchanged = fingerprint_before == fingerprint_after
        payload["integrity"] = {
            "source_database_snapshot_used": True,
            "source_fingerprint_before": fingerprint_before,
            "source_fingerprint_after": fingerprint_after,
            "source_unchanged": unchanged,
            "semantic_vectors_written_to_source_database": False,
            "model_files_cached_outside_database": True,
        }
        if not unchanged:
            raise RuntimeError(
                "La base a changé pendant l’étape 3. Le rapport est abandonné."
            )
        save_semantic_report(original, payload)
        gc.collect()
    return payload
