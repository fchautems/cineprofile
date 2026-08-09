from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import sqlite3
import tempfile
import time
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    ndcg_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from . import __version__
from .db import connect, initialize
from .diagnostics import configure_logging
from . import hybrid_model as hm
from . import personal_model as pm
from .media_types import is_series_type


AUDIT_SCHEMA_VERSION = 3
DEFAULT_HYBRID_VARIANTS = tuple(hm.HYBRID_VARIANTS)
PROMOTABLE_HYBRID_VARIANTS = {
    "structured",
    "structured_lexical",
    "structured_dense",
    "structured_lexical_dense",
}
ProgressCallback = Callable[[int, int, str], None]


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _database_fingerprint(database: str | Path | None) -> str:
    """Build a logical digest of every CineProfile application table."""

    def encode(value: object) -> bytes:
        if value is None:
            return b"N;"
        if isinstance(value, bytes):
            return b"B" + value.hex().encode("ascii") + b";"
        if isinstance(value, float):
            return b"F" + repr(value).encode("ascii") + b";"
        if isinstance(value, int):
            return b"I" + str(value).encode("ascii") + b";"
        return (
            b"T"
            + str(value).encode("utf-8", errors="surrogatepass")
            + b";"
        )

    digest = hashlib.sha256()
    with closing(connect(database)) as connection:
        tables = [
            str(row["name"])
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
        ]
        for table in tables:
            quoted_table = '"' + table.replace('"', '""') + '"'
            columns = [
                str(row["name"])
                for row in connection.execute(
                    f"PRAGMA table_info({quoted_table})"
                ).fetchall()
            ]
            digest.update(b"TABLE:")
            digest.update(table.encode("utf-8"))
            digest.update(b"\n")
            if not columns:
                continue
            order = ", ".join(
                str(index) for index in range(1, len(columns) + 1)
            )
            rows = connection.execute(
                f"SELECT * FROM {quoted_table} ORDER BY {order}"
            ).fetchall()
            digest.update(
                ("COLUMNS:" + "\x1f".join(columns) + "\n").encode("utf-8")
            )
            for row in rows:
                for value in row:
                    digest.update(encode(value))
                digest.update(b"\n")
    return digest.hexdigest()


def _snapshot_database(
    database: str | Path | None,
    destination: Path,
) -> None:
    with closing(connect(database)) as source:
        with closing(sqlite3.connect(destination)) as target:
            source.backup(target)


def _metric_summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"engine", "split", "selected_alpha", "active_engine"}
            and _finite(value) is not None
        }
    )
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray(
            [
                float(row[key])
                for row in rows
                if _finite(row.get(key)) is not None
            ],
            dtype=float,
        )
        if not len(values):
            continue
        summary[key] = {
            "mean": round(float(np.mean(values)), 6),
            "stddev": round(float(np.std(values, ddof=0)), 6),
            "minimum": round(float(np.min(values)), 6),
            "maximum": round(float(np.max(values)), 6),
        }
    return summary


def _metrics(
    scores: np.ndarray,
    ratings: np.ndarray,
    *,
    predicted_ratings: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
) -> dict[str, float | None]:
    scores = np.asarray(scores, dtype=float)
    ratings = np.asarray(ratings, dtype=float)
    labels = (ratings >= pm.LIKE_THRESHOLD).astype(int)
    result: dict[str, float | None] = {
        "auc": (
            float(roc_auc_score(labels, scores))
            if len(np.unique(labels)) >= 2
            else None
        ),
        "positive_rate": float(np.mean(labels)),
    }
    for size in (10, 20, 50):
        k = max(1, min(size, len(scores)))
        indexes = np.argsort(scores)[::-1][:k]
        result[f"precision_at_{size}"] = float(np.mean(labels[indexes]))
        result[f"average_rating_at_{size}"] = float(
            np.mean(ratings[indexes])
        )
        result[f"ndcg_at_{size}"] = float(
            ndcg_score([labels], [scores], k=k)
        )
    if predicted_ratings is not None:
        result["mae"] = float(
            mean_absolute_error(
                ratings,
                np.asarray(predicted_ratings, dtype=float),
            )
        )
    if probabilities is not None:
        clipped = np.clip(
            np.asarray(probabilities, dtype=float),
            0.001,
            0.999,
        )
        result["brier"] = float(brier_score_loss(labels, clipped))
        result["mean_probability"] = float(np.mean(clipped))
        result["calibration_bias"] = float(
            np.mean(clipped) - np.mean(labels)
        )
        result["ece_global"] = _expected_calibration_error(
            clipped,
            labels,
        )
        for size in (10, 20):
            k = max(1, min(size, len(clipped)))
            indexes = np.argsort(scores)[::-1][:k]
            result[f"ece_at_{size}"] = _expected_calibration_error(
                clipped[indexes],
                labels[indexes],
            )
    return result


def _expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 5,
) -> float:
    """Adaptive-bin calibration error, stable even on a top-10 sample."""

    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if not len(probabilities):
        return 0.0
    order = np.argsort(probabilities)
    groups = np.array_split(order, min(bins, len(order)))
    error = 0.0
    for indexes in groups:
        if not len(indexes):
            continue
        weight = len(indexes) / len(probabilities)
        error += weight * abs(
            float(np.mean(probabilities[indexes]))
            - float(np.mean(labels[indexes]))
        )
    return float(error)


def _predict_engine(
    model: pm.PersonalTasteModel,
    test: list[dict],
    engine: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # ``test`` already contains the normalized training-item representation.
    # It must not pass through ``predict_personal_candidate`` because that
    # public function expects a raw TMDB candidate and would discard the
    # normalized entities, benchmark and runtime used by training.
    if engine == "islands_v07":
        if model.islands is None:
            raise ValueError("Le moteur v0.7 n’est pas disponible.")
        predictions = [
            pm.predict_taste_islands(model.islands, item)
            for item in test
        ]
        raw_probabilities = np.asarray(
            [
                float(row["like_probability"]) / 100.0
                for row in predictions
            ],
            dtype=float,
        )
        coverages = np.asarray(
            [float(row["coverage"]) / 100.0 for row in predictions],
            dtype=float,
        )
        separations = np.asarray(
            [
                min(
                    1.0,
                    0.55 + abs(float(row["island_margin"])) / 100.0,
                )
                for row in predictions
            ],
            dtype=float,
        )
        reliabilities = np.asarray(
            [
                float(item.get("benchmark_reliability") or 0.0)
                for item in test
            ],
            dtype=float,
        )
        probabilities = np.asarray(
            [
                pm._conservative_probability(
                    raw,
                    pm._probability_strength(
                        coverage,
                        reliability,
                        separation=separation,
                    ),
                    float(model.metrics["base_like_rate"]),
                )
                for raw, coverage, reliability, separation in zip(
                    raw_probabilities,
                    coverages,
                    reliabilities,
                    separations,
                    strict=True,
                )
            ],
            dtype=float,
        )
        mae = float(model.metrics["islands_v07"]["mae"])
        confidences = np.asarray(
            [
                pm._evidence_confidence(
                    mae,
                    coverage,
                    reliability,
                    separation=separation,
                )
                for coverage, reliability, separation in zip(
                    coverages,
                    reliabilities,
                    separations,
                    strict=True,
                )
            ],
            dtype=float,
        )
        predicted_ratings = np.asarray(
            [float(row["predicted_rating"]) for row in predictions],
            dtype=float,
        )
        return probabilities, predicted_ratings, confidences, coverages

    matrix, text_matrix = pm._transform(model.space, test)
    coverages = np.asarray(
        [
            pm._space_coverage(
                model.space,
                item,
                (
                    text_matrix.getrow(index)
                    if text_matrix is not None
                    else None
                ),
            )
            for index, item in enumerate(test)
        ],
        dtype=float,
    )
    raw_residuals = np.asarray(
        model.regressor.predict(matrix),
        dtype=float,
    )
    benchmarks = np.asarray(
        [
            pm._benchmark(
                item,
                model.user_baseline - model.mean_personal_residual,
            )
            for item in test
        ],
        dtype=float,
    )
    personal_factors = 0.45 + 0.55 * coverages
    residuals = model.mean_personal_residual + (
        raw_residuals - model.mean_personal_residual
    ) * personal_factors
    predicted_ratings = np.clip(benchmarks + residuals, 1.0, 10.0)
    raw_probabilities = model.calibrator.predict_proba(
        predicted_ratings.reshape(-1, 1)
    )[:, 1]
    reliabilities = np.asarray(
        [
            float(item.get("benchmark_reliability") or 0.0)
            for item in test
        ],
        dtype=float,
    )
    probabilities = np.asarray(
        [
            pm._conservative_probability(
                raw,
                pm._probability_strength(coverage, reliability),
                float(model.metrics["base_like_rate"]),
            )
            for raw, coverage, reliability in zip(
                raw_probabilities,
                coverages,
                reliabilities,
                strict=True,
            )
        ],
        dtype=float,
    )
    mae = float(model.metrics["linear_v06"]["mae"])
    confidences = np.asarray(
        [
            pm._evidence_confidence(mae, coverage, reliability)
            for coverage, reliability in zip(
                coverages,
                reliabilities,
                strict=True,
            )
        ],
        dtype=float,
    )
    return probabilities, predicted_ratings, confidences, coverages


def _evaluate_split(
    items: list[dict],
    train_indexes: np.ndarray,
    test_indexes: np.ndarray,
    *,
    split_name: str,
) -> tuple[list[dict], dict]:
    train = [items[int(index)] for index in train_indexes]
    test = [items[int(index)] for index in test_indexes]
    fingerprint = f"audit:{split_name}:{len(train)}:{len(test)}"
    model = pm._train_model(train, fingerprint)
    ratings = np.asarray([float(item["rating"]) for item in test], dtype=float)

    rows: list[dict] = []
    detail: dict[str, object] = {
        "split": split_name,
        "train_count": len(train),
        "test_count": len(test),
        "train_positive_count": sum(
            float(item["rating"]) >= pm.LIKE_THRESHOLD for item in train
        ),
        "test_positive_count": int(
            np.sum(ratings >= pm.LIKE_THRESHOLD)
        ),
        "selected_alpha": float(model.selected_alpha),
        "active_engine_on_training": model.metrics.get("active_engine"),
        "active_selection_reason": model.metrics.get("selection_reason"),
    }

    for engine in ("linear_v06", "islands_v07"):
        if engine == "islands_v07" and model.islands is None:
            continue
        probabilities, predicted, confidence, coverage = _predict_engine(
            model,
            test,
            engine,
        )
        metrics = _metrics(
            probabilities,
            ratings,
            predicted_ratings=predicted,
            probabilities=probabilities,
        )
        metrics.update(
            {
                "engine": engine,
                "split": split_name,
                "selected_alpha": float(
                    model.selected_alpha
                    if engine == "linear_v06"
                    else model.islands.selected_alpha
                ),
                "mean_confidence": float(np.mean(confidence)),
                "mean_coverage": float(np.mean(coverage)),
            }
        )
        rows.append(metrics)

    test_matrix, test_text = pm._transform(model.space, test)
    _, train_text = pm._transform(model.space, train)
    legacy_scores = pm._legacy_scores(
        train,
        test,
        train_text,
        test_text,
    )
    rows.append(
        {
            **_metrics(legacy_scores, ratings),
            "engine": "legacy_v05",
            "split": split_name,
        }
    )

    public_predictions = np.asarray(
        [
            pm._benchmark(item, model.user_baseline)
            + model.mean_personal_residual
            for item in test
        ],
        dtype=float,
    )
    rows.append(
        {
            **_metrics(
                public_predictions,
                ratings,
                predicted_ratings=public_predictions,
            ),
            "engine": "public_baseline",
            "split": split_name,
        }
    )

    active_engine = str(model.metrics.get("active_engine") or "linear_v06")
    active_row = next(
        (
            row
            for row in rows
            if row["engine"] == active_engine
        ),
        None,
    )
    detail["active_outer_metrics"] = active_row
    detail["metrics"] = rows

    if split_name == "chronological":
        examples: dict[str, list[dict]] = {}
        for row in rows:
            engine = str(row["engine"])
            if engine == "linear_v06":
                scores = _predict_engine(model, test, engine)[0]
            elif engine == "islands_v07" and model.islands is not None:
                scores = _predict_engine(model, test, engine)[0]
            elif engine == "legacy_v05":
                scores = legacy_scores
            else:
                scores = public_predictions
            top = np.argsort(scores)[::-1][:20]
            examples[engine] = [
                {
                    "title": test[int(index)]["title"],
                    "date_rated": test[int(index)].get("date_rated"),
                    "actual_rating": float(test[int(index)]["rating"]),
                    "score": round(float(scores[int(index)]), 4),
                }
                for index in top
            ]
        detail["top_20_examples"] = examples
    return rows, detail


def _evaluate_hybrid_variants(
    items: list[dict],
    dense_embeddings: np.ndarray | None,
    train_indexes: np.ndarray,
    test_indexes: np.ndarray,
    *,
    split_name: str,
    variants: tuple[str, ...],
    on_variant: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    train = [items[int(index)] for index in train_indexes]
    test = [items[int(index)] for index in test_indexes]
    train_dense = (
        dense_embeddings[train_indexes]
        if dense_embeddings is not None
        else None
    )
    test_dense = (
        dense_embeddings[test_indexes]
        if dense_embeddings is not None
        else None
    )
    rows: list[dict] = []
    details: dict[str, dict] = {}
    for variant in variants:
        if on_variant:
            on_variant(variant)
        variant_dense_train = (
            train_dense if "dense" in hm.HYBRID_VARIANTS[variant]["blocks"]
            else None
        )
        variant_dense_test = (
            test_dense if "dense" in hm.HYBRID_VARIANTS[variant]["blocks"]
            else None
        )
        metrics, detail = hm.evaluate_hybrid_split(
            train,
            test,
            variant_dense_train,
            variant_dense_test,
            variant=variant,
            split_name=split_name,
        )
        rows.append(metrics)
        details[variant] = detail
    return rows, details


def _evaluate_semantic_retrieval(
    items: list[dict],
    dense_embeddings: np.ndarray,
    train_indexes: np.ndarray,
    test_indexes: np.ndarray,
    *,
    split_name: str,
) -> tuple[dict, dict]:
    """Simulate candidate retrieval using only ratings available in training."""

    train_ratings = np.asarray(
        [float(items[int(index)]["rating"]) for index in train_indexes],
        dtype=float,
    )
    test_ratings = np.asarray(
        [float(items[int(index)]["rating"]) for index in test_indexes],
        dtype=float,
    )
    positive_indexes = train_indexes[train_ratings >= pm.LIKE_THRESHOLD]
    negative_indexes = train_indexes[train_ratings <= 6.0]
    if not len(positive_indexes) or not len(negative_indexes):
        raise ValueError(
            "Pas assez de repères positifs et négatifs pour la récupération."
        )
    test_vectors = dense_embeddings[test_indexes]
    positive_similarities = test_vectors @ dense_embeddings[
        positive_indexes
    ].T
    negative_similarities = test_vectors @ dense_embeddings[
        negative_indexes
    ].T

    def strongest(matrix: np.ndarray, count: int = 3) -> np.ndarray:
        width = min(count, matrix.shape[1])
        return np.mean(
            np.partition(matrix, -width, axis=1)[:, -width:],
            axis=1,
        )

    positive = strongest(positive_similarities)
    negative = strongest(negative_similarities)
    scores = positive - 0.70 * negative
    metrics = {
        **_metrics(scores, test_ratings),
        "engine": "semantic_retrieval_v09",
        "split": split_name,
        "positive_anchor_count": int(len(positive_indexes)),
        "negative_anchor_count": int(len(negative_indexes)),
    }
    labels = test_ratings >= pm.LIKE_THRESHOLD
    order = np.argsort(scores)[::-1]
    for size in (20, 50, 100):
        k = min(size, len(order))
        metrics[f"liked_recall_at_{size}"] = float(
            np.sum(labels[order[:k]]) / max(1, np.sum(labels))
        )
        metrics[f"disliked_share_at_{size}"] = float(
            np.mean(test_ratings[order[:k]] <= 6.0)
        )
    detail = {
        "split": split_name,
        "positive_anchor_count": int(len(positive_indexes)),
        "negative_anchor_count": int(len(negative_indexes)),
        "top_20_examples": [
            {
                "title": items[int(test_indexes[int(index)])]["title"],
                "actual_rating": float(test_ratings[int(index)]),
                "retrieval_score": round(float(scores[int(index)]), 4),
            }
            for index in order[:20]
        ],
    }
    return metrics, detail


def _database_health(
    database: str | Path | None,
    items: list[dict],
) -> dict:
    with closing(connect(database)) as connection:
        rows = connection.execute(
            """
            SELECT title_type, metadata_status, COUNT(*) AS count
            FROM titles
            GROUP BY title_type, metadata_status
            """
        ).fetchall()
        duplicate_tmdb = connection.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT tmdb_id FROM titles
              WHERE tmdb_id IS NOT NULL
              GROUP BY tmdb_id HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        invalid_ratings = connection.execute(
            """
            SELECT COUNT(*) FROM titles
            WHERE user_rating < 1 OR user_rating > 10
            """
        ).fetchone()[0]
        total = connection.execute(
            "SELECT COUNT(*) FROM titles"
        ).fetchone()[0]

    title_types = Counter(
        str(item["title_type"] or "(absent)")
        for item in items
    )
    statuses = Counter(
        str(item["metadata_status"] or "(absent)")
        for item in items
    )
    benchmark_sources = Counter(
        str(item["benchmark_source"] or "(absent)")
        for item in items
    )
    suspicious_tv = {
        str(row["title_type"] or "(absent)"): int(row["count"])
        for row in rows
        if is_series_type(row["title_type"])
    }
    return {
        "database_title_count": int(total),
        "model_item_count": len(items),
        "excluded_by_current_series_rule": int(total - len(items)),
        "title_types_used_by_model": dict(title_types),
        "metadata_statuses_used_by_model": dict(statuses),
        "benchmark_sources": dict(benchmark_sources),
        "missing_public_benchmark": sum(
            item.get("benchmark") is None for item in items
        ),
        "missing_overview": sum(
            not str(item.get("overview") or "").strip()
            for item in items
        ),
        "dated_ratings": sum(bool(item.get("date_rated")) for item in items),
        "invalid_user_ratings": int(invalid_ratings),
        "duplicate_tmdb_ids": int(duplicate_tmdb),
        "series_types_detected_in_source": suspicious_tv,
        "series_types_still_used_by_model": {
            name: count
            for name, count in title_types.items()
            if is_series_type(name)
        },
        "suspicious_tv_types_not_excluded_by_current_rule": {
            name: count
            for name, count in title_types.items()
            if is_series_type(name)
        },
        "raw_type_status_counts": [
            {
                "title_type": row["title_type"],
                "metadata_status": row["metadata_status"],
                "count": int(row["count"]),
            }
            for row in rows
        ],
    }


def _learning_sizes(maximum: int) -> list[int]:
    proposed = [400, 700, 1000, maximum]
    return sorted(
        {
            max(pm.MINIMUM_RATINGS, min(maximum, value))
            for value in proposed
            if maximum >= pm.MINIMUM_RATINGS
        }
    )


def _summary_mean(
    summaries: dict[str, dict[str, dict[str, float]]],
    engine: str,
    metric: str,
) -> float | None:
    return _finite(
        summaries.get(engine, {}).get(metric, {}).get("mean")
    )


def _metric_winner(
    summaries: dict[str, dict[str, dict[str, float]]],
    metric: str,
    *,
    lower_is_better: bool = False,
) -> dict | None:
    values = [
        (engine, value)
        for engine in summaries
        if (
            value := _summary_mean(summaries, engine, metric)
        ) is not None
    ]
    if not values:
        return None
    engine, value = (
        min(values, key=lambda row: row[1])
        if lower_is_better
        else max(values, key=lambda row: row[1])
    )
    return {
        "engine": engine,
        "value": round(float(value), 6),
    }


def _learning_curve_conclusion(
    learning_summary: dict[str, dict],
) -> dict:
    sizes = sorted(int(size) for size in learning_summary)
    if not sizes:
        return {
            "largest_tested_size": None,
            "smallest_comparable_size": None,
            "interpretation": "Courbe d’apprentissage indisponible.",
        }
    largest = sizes[-1]

    def mean(size: int, metric: str) -> float | None:
        return _finite(
            learning_summary[str(size)].get(metric, {}).get("mean")
        )

    reference_precision = mean(largest, "precision_at_20")
    reference_auc = mean(largest, "auc")
    reference_mae = mean(largest, "mae")
    comparable: list[int] = []
    for size in sizes:
        precision = mean(size, "precision_at_20")
        auc = mean(size, "auc")
        mae = mean(size, "mae")
        checks = []
        if reference_precision is not None and precision is not None:
            checks.append(precision >= reference_precision - 0.025)
        if reference_auc is not None and auc is not None:
            checks.append(auc >= reference_auc - 0.010)
        if reference_mae is not None and mae is not None:
            checks.append(mae <= reference_mae + 0.050)
        if checks and all(checks):
            comparable.append(size)
    smallest = min(comparable) if comparable else largest
    interpretation = (
        f"{smallest} notes suffisent à atteindre pratiquement le niveau "
        f"mesuré avec {largest} notes."
        if smallest < largest
        else (
            f"La performance progresse encore jusqu’à {largest} notes ; "
            "réduire arbitrairement l’apprentissage n’est pas conseillé."
        )
    )
    return {
        "largest_tested_size": largest,
        "smallest_comparable_size": smallest,
        "tolerances": {
            "precision_at_20": 0.025,
            "auc": 0.010,
            "mae": 0.050,
        },
        "interpretation": interpretation,
    }


def _optimizer_recommendation(
    engine_summaries: dict[str, dict[str, dict[str, float]]],
    random_details: list[dict],
    chronological: dict | None,
) -> dict:
    """Choose a challenger conservatively; never blend two engines.

    NDCG@10 is the primary objective because the useful screen is the first
    handful of suggestions.  The remaining gates prevent a narrow top-10 gain
    from hiding worse probability quality, rating error or recent-history
    behavior.
    """

    baseline_engine = "linear_v06"
    baseline = engine_summaries.get(baseline_engine)
    candidates = [
        engine
        for engine in engine_summaries
        if engine.startswith("personal_")
        and engine.removeprefix("personal_").removesuffix("_v09")
        in PROMOTABLE_HYBRID_VARIANTS
    ]
    if not baseline or not candidates:
        return {
            "decision": "keep_default",
            "baseline_engine": baseline_engine,
            "candidate_engine": None,
            "variant": None,
            "message": (
                "Aucune variante personnelle complète n’a pu être comparée ; "
                "la configuration personnelle v0.9 par défaut reste active."
            ),
            "gates": {},
        }

    def value(engine: str, metric: str, fallback: float) -> float:
        result = _summary_mean(engine_summaries, engine, metric)
        return fallback if result is None else float(result)

    candidate = max(
        candidates,
        key=lambda engine: (
            value(engine, "ndcg_at_10", -1.0),
            value(engine, "precision_at_10", -1.0),
            value(engine, "ndcg_at_20", -1.0),
            -value(engine, "mae", 99.0),
        ),
    )
    variant = candidate.removeprefix("personal_").removesuffix("_v09")
    ndcg_gain = value(candidate, "ndcg_at_10", 0.0) - value(
        baseline_engine,
        "ndcg_at_10",
        0.0,
    )
    precision_gap = value(candidate, "precision_at_10", 0.0) - value(
        baseline_engine,
        "precision_at_10",
        0.0,
    )
    auc_gap = value(candidate, "auc", 0.0) - value(
        baseline_engine,
        "auc",
        0.0,
    )
    mae_gap = value(candidate, "mae", 99.0) - value(
        baseline_engine,
        "mae",
        99.0,
    )
    brier_gap = value(candidate, "brier", 99.0) - value(
        baseline_engine,
        "brier",
        99.0,
    )

    split_wins = 0
    split_comparisons = 0
    selected_alphas: list[float] = []
    for detail in random_details:
        rows = {
            str(row.get("engine")): row
            for row in detail.get("metrics", [])
        }
        baseline_row = rows.get(baseline_engine)
        candidate_row = rows.get(candidate)
        if not baseline_row or not candidate_row:
            continue
        hybrid_detail = detail.get("hybrid", {}).get(variant, {})
        if _finite(hybrid_detail.get("selected_alpha")) is not None:
            selected_alphas.append(
                float(hybrid_detail["selected_alpha"])
            )
        split_comparisons += 1
        if float(candidate_row.get("ndcg_at_10") or 0.0) > float(
            baseline_row.get("ndcg_at_10") or 0.0
        ):
            split_wins += 1
    split_win_rate = split_wins / max(1, split_comparisons)
    alpha_counts = Counter(selected_alphas)
    selected_alpha = (
        max(
            alpha_counts,
            key=lambda alpha: (alpha_counts[alpha], -alpha),
        )
        if alpha_counts
        else None
    )
    boundary_rate = (
        sum(alpha == max(hm.HYBRID_ALPHAS) for alpha in selected_alphas)
        / max(1, len(selected_alphas))
    )

    chronological_ndcg_gap: float | None = None
    chronological_precision_gap: float | None = None
    if chronological:
        rows = {
            str(row.get("engine")): row
            for row in chronological.get("metrics", [])
        }
        baseline_row = rows.get(baseline_engine)
        candidate_row = rows.get(candidate)
        if baseline_row and candidate_row:
            chronological_ndcg_gap = float(
                candidate_row.get("ndcg_at_10") or 0.0
            ) - float(baseline_row.get("ndcg_at_10") or 0.0)
            chronological_precision_gap = float(
                candidate_row.get("precision_at_10") or 0.0
            ) - float(baseline_row.get("precision_at_10") or 0.0)

    gates = {
        "meaningful_top10_gain": {
            "passed": ndcg_gain >= 0.015,
            "observed": round(ndcg_gain, 6),
            "minimum": 0.015,
        },
        "top10_precision_preserved": {
            "passed": precision_gap >= -0.02,
            "observed": round(precision_gap, 6),
            "minimum": -0.02,
        },
        "global_ranking_preserved": {
            "passed": auc_gap >= -0.01,
            "observed": round(auc_gap, 6),
            "minimum": -0.01,
        },
        "rating_error_preserved": {
            "passed": mae_gap <= 0.05,
            "observed": round(mae_gap, 6),
            "maximum": 0.05,
        },
        "probability_error_preserved": {
            "passed": brier_gap <= 0.03,
            "observed": round(brier_gap, 6),
            "maximum": 0.03,
        },
        "repeatability": {
            "passed": split_comparisons >= 2 and split_win_rate >= 0.60,
            "observed": round(split_win_rate, 6),
            "minimum": 0.60,
            "wins": split_wins,
            "comparisons": split_comparisons,
        },
        "regularization_search_has_margin": {
            "passed": boundary_rate < 0.50,
            "observed": round(boundary_rate, 6),
            "maximum": 0.50,
            "selected_alpha": selected_alpha,
        },
    }
    if chronological_ndcg_gap is not None:
        gates["recent_history_preserved"] = {
            "passed": (
                chronological_ndcg_gap >= -0.03
                and float(chronological_precision_gap or 0.0) >= -0.05
            ),
            "observed_ndcg": round(chronological_ndcg_gap, 6),
            "observed_precision": round(
                float(chronological_precision_gap or 0.0),
                6,
            ),
            "minimum_ndcg": -0.03,
            "minimum_precision": -0.05,
        }
    passed = all(bool(row["passed"]) for row in gates.values())
    label = hm.HYBRID_VARIANTS[variant]["label"]
    return {
        "decision": "promote" if passed else "keep_default",
        "baseline_engine": baseline_engine,
        "candidate_engine": candidate,
        "variant": variant,
        "variant_label": label,
        "selected_alpha": selected_alpha,
        "selected_alpha_counts": {
            f"{alpha:g}": count
            for alpha, count in sorted(alpha_counts.items())
        },
        "gates": gates,
        "message": (
            f"Le challenger « {label} » passe tous les contrôles. Il peut "
            "être activé sans mélanger son score à la v0.6."
            if passed
            else (
                f"La variante testée la plus forte (« {label} ») ne passe pas "
                "encore tous les contrôles ; la configuration personnelle "
                "v0.9 par défaut reste active."
            )
        ),
    }


def _automated_findings(
    engine_summaries: dict[str, dict[str, dict[str, float]]],
    learning_summary: dict[str, dict],
    chronological: dict | None,
    database_health: dict,
    random_details: list[dict],
) -> dict:
    winners = {
        "precision_at_20": _metric_winner(
            engine_summaries, "precision_at_20"
        ),
        "ndcg_at_20": _metric_winner(engine_summaries, "ndcg_at_20"),
        "auc": _metric_winner(engine_summaries, "auc"),
        "mae": _metric_winner(
            engine_summaries, "mae", lower_is_better=True
        ),
        "brier": _metric_winner(
            engine_summaries, "brier", lower_is_better=True
        ),
    }
    learned_engines = [
        engine
        for engine in ("linear_v06", "islands_v07")
        if engine in engine_summaries
    ]
    public_precision = _summary_mean(
        engine_summaries,
        "public_baseline",
        "precision_at_20",
    )
    learned_precisions = [
        value
        for engine in learned_engines
        if (
            value := _summary_mean(
                engine_summaries, engine, "precision_at_20"
            )
        )
        is not None
    ]
    public_dominance = (
        public_precision is not None
        and learned_precisions
        and public_precision >= max(learned_precisions) + 0.05
    )
    chronological_winners: dict[str, dict | None] | None = None
    if chronological:
        chronological_summaries = {
            str(row["engine"]): {
                key: {"mean": value}
                for key, value in row.items()
                if _finite(value) is not None
            }
            for row in chronological.get("metrics", [])
        }
        chronological_winners = {
            "precision_at_20": _metric_winner(
                chronological_summaries,
                "precision_at_20",
            ),
            "ndcg_at_20": _metric_winner(
                chronological_summaries,
                "ndcg_at_20",
            ),
            "auc": _metric_winner(
                chronological_summaries,
                "auc",
            ),
            "mae": _metric_winner(
                chronological_summaries,
                "mae",
                lower_is_better=True,
            ),
        }
    warnings: list[dict[str, str]] = []
    if public_dominance:
        warnings.append(
            {
                "code": "public_baseline_beats_personal_top20",
                "message": (
                    "La note publique seule retrouve au moins 5 points de "
                    "pourcentage de films appréciés de plus que les moteurs "
                    "personnels dans le top 20."
                ),
            }
        )
    if database_health.get("suspicious_tv_types_not_excluded_by_current_rule"):
        warnings.append(
            {
                "code": "tv_type_leakage",
                "message": (
                    "Des titres de type TV ne sont pas exclus par la règle "
                    "actuelle des films."
                ),
            }
        )
    if int(database_health.get("invalid_user_ratings") or 0):
        warnings.append(
            {
                "code": "invalid_user_ratings",
                "message": "Certaines notes IMDb sont hors de l’intervalle 1–10.",
            }
        )
    item_count = max(1, int(database_health.get("model_item_count") or 0))
    if int(database_health.get("missing_overview") or 0) / item_count > 0.10:
        warnings.append(
            {
                "code": "missing_overviews",
                "message": (
                    "Plus de 10 % des films du modèle n’ont pas de résumé ; "
                    "les signaux textuels sont donc incomplets."
                ),
            }
        )
    return {
        "random_holdout_winners": winners,
        "chronological_winners": chronological_winners,
        "public_baseline_materially_better_at_top20": bool(public_dominance),
        "learning_curve": _learning_curve_conclusion(learning_summary),
        "optimizer_recommendation": _optimizer_recommendation(
            engine_summaries,
            random_details,
            chronological,
        ),
        "warnings": warnings,
    }


def _run_backtest_audit_snapshot(
    database: str | Path | None = None,
    *,
    repeats: int = 5,
    test_fraction: float = 0.25,
    learning_curve_repeats: int = 2,
    hybrid_variants: tuple[str, ...] = (),
    semantic_preparation_error: str | None = None,
    on_progress: ProgressCallback | None = None,
    logger: logging.Logger,
) -> dict:
    initialize(database)
    started = time.perf_counter()
    items = pm._load_training_items(database)
    if len(items) < 120:
        raise ValueError(
            "L’audit répété demande au moins 120 films exploitables."
        )
    ratings = np.asarray([float(item["rating"]) for item in items], dtype=float)
    labels = (ratings >= pm.LIKE_THRESHOLD).astype(int)
    if min(int(labels.sum()), int(len(labels) - labels.sum())) < 20:
        raise ValueError(
            "Les notes ne contiennent pas assez d’exemples contrastés "
            "pour un audit répété."
        )
    repeats = max(2, min(20, int(repeats)))
    test_fraction = max(0.15, min(0.40, float(test_fraction)))
    curve_repeats = max(0, min(repeats, int(learning_curve_repeats)))
    variants = tuple(
        variant
        for variant in hybrid_variants
        if variant in hm.HYBRID_VARIANTS
    )
    dense_embeddings = (
        hm.load_cached_dense_embeddings(database, items)
        if any(
            "dense" in hm.HYBRID_VARIANTS[variant]["blocks"]
            for variant in variants
        )
        else None
    )
    random_splitter = StratifiedShuffleSplit(
        n_splits=repeats,
        test_size=test_fraction,
        random_state=19791101,
    )
    splits = list(
        random_splitter.split(np.zeros(len(items)), labels)
    )
    maximum_train = len(splits[0][0])
    learning_sizes = _learning_sizes(maximum_train)
    curve_jobs = curve_repeats * max(0, len(learning_sizes) - 1)
    dated = [
        (index, item)
        for index, item in enumerate(items)
        if str(item.get("date_rated") or "")
        and len(str(item.get("date_rated"))) >= 10
    ]
    chronological_available = len(dated) >= int(0.70 * len(items))
    engines_per_split = 1 + len(variants)
    total_jobs = (
        repeats * engines_per_split
        + curve_jobs
        + int(chronological_available) * engines_per_split
        + (repeats + int(chronological_available))
        * int(dense_embeddings is not None)
    )
    current_job = 0
    random_rows: list[dict] = []
    random_details: list[dict] = []
    logger.info(
        "audit_started | items=%s | repeats=%s | test_fraction=%.3f | "
        "learning_sizes=%s",
        len(items),
        repeats,
        test_fraction,
        learning_sizes,
    )

    def progress(message: str) -> None:
        if on_progress:
            on_progress(current_job, total_jobs, message)

    for split_index, (train_indexes, test_indexes) in enumerate(
        splits,
        start=1,
    ):
        progress(
            f"Backtest aléatoire {split_index}/{repeats}"
        )
        split_started = time.perf_counter()
        rows, detail = _evaluate_split(
            items,
            train_indexes,
            test_indexes,
            split_name=f"random_{split_index:02d}",
        )
        current_job += 1

        def random_variant_progress(variant: str) -> None:
            nonlocal current_job
            progress(
                "Challenger "
                + str(hm.HYBRID_VARIANTS[variant]["label"])
                + f" · découpage {split_index}/{repeats}"
            )
            current_job += 1

        hybrid_rows, hybrid_details = _evaluate_hybrid_variants(
            items,
            dense_embeddings,
            train_indexes,
            test_indexes,
            split_name=f"random_{split_index:02d}",
            variants=variants,
            on_variant=random_variant_progress,
        )
        rows.extend(hybrid_rows)
        detail["hybrid"] = hybrid_details
        if dense_embeddings is not None:
            progress(
                f"Récupération personnalisée · découpage "
                f"{split_index}/{repeats}"
            )
            retrieval_row, retrieval_detail = _evaluate_semantic_retrieval(
                items,
                dense_embeddings,
                train_indexes,
                test_indexes,
                split_name=f"random_{split_index:02d}",
            )
            rows.append(retrieval_row)
            detail["semantic_retrieval"] = retrieval_detail
            current_job += 1
        detail["metrics"] = rows
        random_rows.extend(rows)
        random_details.append(detail)
        for row in rows:
            logger.info(
                "audit_metrics | split=random_%02d | engine=%s | "
                "auc=%s | precision_at_20=%s | ndcg_at_20=%s | "
                "mae=%s | brier=%s",
                split_index,
                row.get("engine"),
                row.get("auc"),
                row.get("precision_at_20"),
                row.get("ndcg_at_20"),
                row.get("mae"),
                row.get("brier"),
            )
        logger.info(
            "audit_split_completed | split=random_%02d | train=%s | "
            "test=%s | elapsed_seconds=%.2f | active=%s",
            split_index,
            len(train_indexes),
            len(test_indexes),
            time.perf_counter() - split_started,
            detail.get("active_engine_on_training"),
        )

    learning_rows: list[dict] = []
    if curve_repeats:
        for split_index, (outer_train, outer_test) in enumerate(
            splits[:curve_repeats],
            start=1,
        ):
            outer_labels = labels[outer_train]
            for train_size in learning_sizes:
                if train_size == len(outer_train):
                    matching = [
                        row
                        for row in random_rows
                        if row["split"] == f"random_{split_index:02d}"
                        and row["engine"] == "linear_v06"
                    ]
                    if matching:
                        learning_rows.append(
                            {
                                **matching[0],
                                "learning_repeat": split_index,
                                "train_size": train_size,
                            }
                        )
                    continue
                progress(
                    f"Courbe d’apprentissage : {train_size} films"
                )
                subset_splitter = StratifiedShuffleSplit(
                    n_splits=1,
                    train_size=train_size,
                    random_state=31000 + split_index * 100 + train_size,
                )
                local_indexes, _ = next(
                    subset_splitter.split(
                        np.zeros(len(outer_train)),
                        outer_labels,
                    )
                )
                subset = outer_train[local_indexes]
                rows, _ = _evaluate_split(
                    items,
                    subset,
                    outer_test,
                    split_name=(
                        f"learning_{split_index:02d}_{train_size}"
                    ),
                )
                linear = next(
                    row for row in rows if row["engine"] == "linear_v06"
                )
                learning_rows.append(
                    {
                        **linear,
                        "learning_repeat": split_index,
                        "train_size": train_size,
                    }
                )
                current_job += 1
                logger.info(
                    "audit_learning_curve | repeat=%s | train=%s | "
                    "test=%s | auc=%s | precision_at_20=%s | mae=%s",
                    split_index,
                    train_size,
                    len(outer_test),
                    linear.get("auc"),
                    linear.get("precision_at_20"),
                    linear.get("mae"),
                )

    chronological: dict | None = None
    if chronological_available:
        progress("Contrôle chronologique")
        dated.sort(
            key=lambda row: (
                str(row[1].get("date_rated") or ""),
                str(row[1].get("id") or ""),
            )
        )
        cut = max(
            pm.MINIMUM_RATINGS,
            int(0.80 * len(dated)),
        )
        chronological_train = np.asarray(
            [index for index, _ in dated[:cut]],
            dtype=int,
        )
        chronological_test = np.asarray(
            [index for index, _ in dated[cut:]],
            dtype=int,
        )
        rows, detail = _evaluate_split(
            items,
            chronological_train,
            chronological_test,
            split_name="chronological",
        )
        current_job += 1

        def chronological_variant_progress(variant: str) -> None:
            nonlocal current_job
            progress(
                "Contrôle récent · "
                + str(hm.HYBRID_VARIANTS[variant]["label"])
            )
            current_job += 1

        hybrid_rows, hybrid_details = _evaluate_hybrid_variants(
            items,
            dense_embeddings,
            chronological_train,
            chronological_test,
            split_name="chronological",
            variants=variants,
            on_variant=chronological_variant_progress,
        )
        rows.extend(hybrid_rows)
        detail["hybrid"] = hybrid_details
        if dense_embeddings is not None:
            progress("Contrôle récent · récupération personnalisée")
            retrieval_row, retrieval_detail = _evaluate_semantic_retrieval(
                items,
                dense_embeddings,
                chronological_train,
                chronological_test,
                split_name="chronological",
            )
            rows.append(retrieval_row)
            detail["semantic_retrieval"] = retrieval_detail
            current_job += 1
        detail["metrics"] = rows
        examples = detail.setdefault("top_20_examples", {})
        for variant, hybrid_detail in hybrid_details.items():
            hybrid_examples = hybrid_detail.get("top_20_examples")
            if hybrid_examples:
                examples[f"personal_{variant}_v09"] = hybrid_examples
        chronological = {
            "metrics": rows,
            "detail": detail,
        }
        for row in rows:
            logger.info(
                "audit_metrics | split=chronological | engine=%s | "
                "auc=%s | precision_at_20=%s | ndcg_at_20=%s | "
                "mae=%s | brier=%s",
                row.get("engine"),
                row.get("auc"),
                row.get("precision_at_20"),
                row.get("ndcg_at_20"),
                row.get("mae"),
                row.get("brier"),
            )

    grouped: dict[str, list[dict]] = {}
    for row in random_rows:
        grouped.setdefault(str(row["engine"]), []).append(row)
    active_counts = Counter(
        str(row.get("active_engine_on_training"))
        for row in random_details
    )
    alpha_counts = Counter(
        str(row.get("selected_alpha"))
        for row in random_details
    )
    learning_summary: dict[str, dict] = {}
    for train_size in learning_sizes:
        selected = [
            row
            for row in learning_rows
            if int(row["train_size"]) == train_size
        ]
        if selected:
            learning_summary[str(train_size)] = _metric_summary(selected)
    engine_summaries = {
        engine: _metric_summary(rows)
        for engine, rows in grouped.items()
    }
    database_health = _database_health(database, items)

    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "app_version": __version__,
        "purpose": (
            "Audit hors échantillon de la version existante ; aucun réglage "
            "du moteur ni aucune note n’est modifié."
        ),
        "methodology": {
            "random_holdout": (
                f"{repeats} découpages stratifiés indépendants, "
                f"{100 * (1 - test_fraction):.0f}% apprentissage et "
                f"{100 * test_fraction:.0f}% test totalement caché"
            ),
            "nested_training": (
                "Chaque modèle est construit et calibré uniquement avec le "
                "groupe d’apprentissage ; le groupe test n’intervient jamais "
                "dans le choix du moteur ou de l’alpha."
            ),
            "challenger_ablation": (
                "Les métadonnées, la syntaxe TF-IDF et la sémantique profonde "
                "sont testées seules puis combinées. Leurs scores ne sont "
                "jamais additionnés après coup."
            ),
            "candidate_retrieval": (
                "Chaque découpage masque aussi des films appréciés puis mesure "
                "si les voisins sémantiques des seuls films d’apprentissage "
                "les font remonter, sans utiliser leur note cachée."
            ),
            "learning_curve": (
                "Plusieurs tailles d’apprentissage sont comparées sur les "
                "mêmes groupes tests."
            ),
            "chronological": (
                "Lorsque les dates sont disponibles, les 20% de notes les "
                "plus récentes sont gardées pour un contrôle final."
            ),
            "like_definition": f"note IMDb ≥ {pm.LIKE_THRESHOLD:g}/10",
            "hybrid_variants": list(variants),
            "semantic_model": (
                hm.SEMANTIC_MODEL
                if any(
                    "dense" in hm.HYBRID_VARIANTS[variant]["blocks"]
                    for variant in variants
                )
                else None
            ),
        },
        "database_health": database_health,
        "random_holdouts": {
            "repeats": repeats,
            "test_fraction": test_fraction,
            "engine_summaries": engine_summaries,
            "active_engine_selection_counts": dict(active_counts),
            "selected_alpha_counts": dict(alpha_counts),
            "splits": random_details,
        },
        "candidate_retrieval": {
            "engine": "semantic_retrieval_v09",
            "summary": engine_summaries.get(
                "semantic_retrieval_v09",
                {},
            ),
            "interpretation": (
                "Le rappel mesure la part des films appréciés masqués retrouvée "
                "dans les premiers candidats ; la part de rejets contrôle les "
                "contre-exemples introduits par cette récupération."
            ),
        },
        "learning_curve": {
            "repeats": curve_repeats,
            "train_sizes": learning_sizes,
            "summaries": learning_summary,
            "runs": learning_rows,
        },
        "chronological": chronological,
        "automated_findings": _automated_findings(
            engine_summaries,
            learning_summary,
            chronological,
            database_health,
            random_details,
        ),
        "semantic_preparation_error": semantic_preparation_error,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "privacy": (
            "Le jeton TMDB, les résumés complets et l’historique intégral ne "
            "sont pas exportés. Seuls les vingt premiers exemples par moteur "
            "du contrôle chronologique sont inclus pour interpréter les "
            "résultats."
        ),
    }
    logger.info(
        "audit_completed | items=%s | elapsed_seconds=%.2f | "
        "active_counts=%s | optimizer=%s",
        len(items),
        payload["elapsed_seconds"],
        json.dumps(dict(active_counts), ensure_ascii=False, sort_keys=True),
        json.dumps(
            payload["automated_findings"]["optimizer_recommendation"],
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    return payload


def run_backtest_audit(
    database: str | Path | None = None,
    *,
    repeats: int = 5,
    test_fraction: float = 0.25,
    learning_curve_repeats: int = 2,
    hybrid_variants: tuple[str, ...] | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict:
    original = Path(database or "data/cineprofile.db")
    initialize(original)
    logger = configure_logging(original)
    requested_variants = tuple(
        DEFAULT_HYBRID_VARIANTS
        if hybrid_variants is None
        else hybrid_variants
    )
    unknown = [
        variant
        for variant in requested_variants
        if variant not in hm.HYBRID_VARIANTS
    ]
    if unknown:
        raise ValueError(
            "Variantes de challenger inconnues : " + ", ".join(unknown)
        )
    semantic_error: str | None = None
    available_variants = requested_variants
    if any(
        "dense" in hm.HYBRID_VARIANTS[variant]["blocks"]
        for variant in requested_variants
    ):
        source_items = pm._load_training_items(original)
        try:
            if on_progress:
                on_progress(
                    0,
                    1,
                    "Préparation locale de la sémantique profonde",
                )
            hm.prepare_dense_embeddings(original, source_items)
        except Exception as exc:
            semantic_error = str(exc)
            available_variants = tuple(
                variant
                for variant in requested_variants
                if "dense" not in hm.HYBRID_VARIANTS[variant]["blocks"]
            )
            logger.exception(
                "audit_semantic_preparation_failed | dense_variants_skipped=%s",
                len(requested_variants) - len(available_variants),
            )
    fingerprint_before = _database_fingerprint(original)
    with tempfile.TemporaryDirectory(
        prefix="cineprofile-audit-",
        ignore_cleanup_errors=True,
    ) as directory:
        snapshot = Path(directory) / "cineprofile-audit.db"
        _snapshot_database(original, snapshot)
        payload = _run_backtest_audit_snapshot(
            snapshot,
            repeats=repeats,
            test_fraction=test_fraction,
            learning_curve_repeats=learning_curve_repeats,
            hybrid_variants=available_variants,
            semantic_preparation_error=semantic_error,
            on_progress=on_progress,
            logger=logger,
        )
        fingerprint_after = _database_fingerprint(original)
        unchanged = fingerprint_before == fingerprint_after
        payload["integrity"] = {
            "audit_used_database_snapshot": True,
            "source_fingerprint_before": fingerprint_before,
            "source_fingerprint_after": fingerprint_after,
            "source_unchanged": unchanged,
        }
        if not unchanged:
            logger.error(
                "audit_integrity_warning | source database changed while the "
                "audit was running"
            )
            raise RuntimeError(
                "La base a changé pendant l’audit. Le résultat est abandonné "
                "pour éviter une comparaison incohérente ; relance l’audit "
                "sans modifier les notes ou les préférences en parallèle."
            )
        # Sauvegarder avant de quitter le dossier temporaire : même si Windows
        # garde brièvement un handle SQLite ouvert, les douze étapes terminées
        # et leur rapport ne peuvent plus être perdus au nettoyage.
        save_backtest_audit(original, payload)
        gc.collect()
    return payload


def save_backtest_audit(
    database: str | Path | None,
    payload: dict,
) -> Path:
    path = backtest_audit_path(database, payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    logging.getLogger("cineprofile").info(
        "audit_saved | file=%s",
        path.name,
    )
    return path


def backtest_audit_path(
    database: str | Path | None,
    payload: dict,
) -> Path:
    target = Path(database or "data/cineprofile.db").parent / "logs"
    target.mkdir(parents=True, exist_ok=True)
    stamp = str(payload["created_at"]).replace(":", "").replace("-", "")
    stamp = stamp.replace("+0000", "Z").replace("+00:00", "Z")
    return target / f"audit_backtest_{stamp}.json"
