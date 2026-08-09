from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    ndcg_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .db import connect, initialize, transaction
from .diagnostics import configure_logging
from . import personal_model as pm
from .semantic import (
    SEMANTIC_MODEL,
    EmbeddingProgressCallback,
    cached_text_embeddings,
    embedding_cache_coverage,
)


HYBRID_MODEL_VERSION = "cineprofile-personal-0.9.0"
DEFAULT_PERSONAL_VARIANT = "structured"
HYBRID_ALPHAS = (10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0)
HYBRID_VARIANTS = {
    "structured": {
        "label": "Profil personnel structuré",
        "blocks": ("structured",),
    },
    "lexical": {
        "label": "Syntaxe TF-IDF",
        "blocks": ("lexical",),
    },
    "structured_lexical": {
        "label": "Métadonnées + syntaxe",
        "blocks": ("structured", "lexical"),
    },
    "dense": {
        "label": "Profil sémantique personnel",
        "blocks": ("dense",),
    },
    "structured_dense": {
        "label": "Profil personnel + sémantique",
        "blocks": ("structured", "dense"),
    },
    "structured_lexical_dense": {
        "label": "Métadonnées + syntaxe + sémantique",
        "blocks": ("structured", "lexical", "dense"),
    },
}


@dataclass
class HybridFeatureSpace:
    variant: str
    structured: DictVectorizer | None
    lexical: TfidfVectorizer | None
    dense_model_name: str | None


@dataclass
class HybridTasteModel:
    version: str
    variant: str
    fingerprint: str
    trained_at: str
    rated_count: int
    selected_alpha: float
    space: HybridFeatureSpace
    regressor: Ridge
    calibrator: LogisticRegression
    user_baseline: float
    mean_personal_residual: float
    base_like_rate: float
    error_p80: float
    metrics: dict


@dataclass
class HybridModelState:
    status: str
    summary: dict
    model: HybridTasteModel | None = None


def _blocks(variant: str) -> tuple[str, ...]:
    try:
        return tuple(HYBRID_VARIANTS[variant]["blocks"])
    except KeyError as exc:
        raise ValueError(f"Variante hybride inconnue : {variant}") from exc


def _model_text(item: dict) -> str:
    text = pm._item_text(item).strip()
    if text:
        return text
    return "Titre : " + str(item.get("title") or "œuvre sans description")


def _personal_structured_features(item: dict) -> dict[str, float]:
    """Keep taste signals and remove public-popularity shortcuts."""

    return {
        name: value
        for name, value in pm._structured_features(item).items()
        if name
        not in {
            "num::public_rating",
            "num::public_missing",
            "num::public_reliability",
            "num::log_votes",
        }
        and not name.startswith("public_source::")
    }


def _embedding_documents(
    items: Iterable[dict],
    *,
    kind: str,
) -> list[dict]:
    return [
        {
            "kind": kind,
            "id": str(item.get("id", "")),
            "text": _model_text(item),
        }
        for item in items
    ]


def prepare_dense_embeddings(
    database: str | Path | None,
    items: list[dict],
    *,
    kind: str = "title",
    cache_directory: str | Path | None = None,
    on_progress: EmbeddingProgressCallback | None = None,
) -> np.ndarray:
    return cached_text_embeddings(
        database,
        _embedding_documents(items, kind=kind),
        model_name=SEMANTIC_MODEL,
        cache_directory=cache_directory,
        on_progress=on_progress,
    )


def dense_embedding_cache_coverage(
    database: str | Path | None,
    items: list[dict],
    *,
    kind: str = "title",
) -> dict[str, int]:
    return embedding_cache_coverage(
        database,
        _embedding_documents(items, kind=kind),
        model_name=SEMANTIC_MODEL,
    )


def load_cached_dense_embeddings(
    database: str | Path | None,
    items: list[dict],
    *,
    kind: str = "title",
) -> np.ndarray:
    """Load vectors and fail if the semantic cache is incomplete.

    Audits call this on a temporary SQLite snapshot.  The source database is
    prepared before the snapshot, so an audit never downloads a 220 MB model
    into its temporary Windows directory.
    """

    documents = _embedding_documents(items, kind=kind)
    if not documents:
        return np.empty((0, 0), dtype=np.float32)
    wanted = {
        (row["kind"], str(row["id"])): hashlib.sha256(
            row["text"].encode("utf-8")
        ).hexdigest()
        for row in documents
    }
    cached: dict[tuple[str, str], np.ndarray] = {}
    with connect(database) as connection:
        rows = connection.execute(
            """
            SELECT item_kind, item_id, text_hash, dimensions, vector_blob
            FROM text_embeddings
            WHERE model_name=?
            """,
            (SEMANTIC_MODEL,),
        ).fetchall()
    for row in rows:
        key = (str(row["item_kind"]), str(row["item_id"]))
        if wanted.get(key) != str(row["text_hash"]):
            continue
        vector = np.frombuffer(row["vector_blob"], dtype=np.float32)
        if len(vector) == int(row["dimensions"]):
            cached[key] = vector
    missing = [
        key for key in wanted if key not in cached
    ]
    if missing:
        raise ValueError(
            f"Le cache sémantique est incomplet ({len(missing)} vecteurs)."
        )
    matrix = np.vstack(
        [cached[(row["kind"], str(row["id"]))] for row in documents]
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(matrix / norms, dtype=np.float32)


def _fit_space(items: list[dict], variant: str) -> HybridFeatureSpace:
    blocks = _blocks(variant)
    structured: DictVectorizer | None = None
    lexical: TfidfVectorizer | None = None
    if "structured" in blocks:
        structured = DictVectorizer(sparse=True, dtype=np.float64)
        structured.fit([_personal_structured_features(item) for item in items])
    if "lexical" in blocks:
        lexical = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=7000,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        try:
            lexical.fit([_model_text(item) for item in items])
        except ValueError:
            lexical = None
    return HybridFeatureSpace(
        variant=variant,
        structured=structured,
        lexical=lexical,
        dense_model_name=SEMANTIC_MODEL if "dense" in blocks else None,
    )


def _transform(
    space: HybridFeatureSpace,
    items: list[dict],
    dense_embeddings: np.ndarray | None,
) -> tuple[csr_matrix, dict[str, csr_matrix | None]]:
    parts: list[csr_matrix] = []
    block_rows: dict[str, csr_matrix | None] = {
        "structured": None,
        "lexical": None,
        "dense": None,
    }
    if space.structured is not None:
        structured = space.structured.transform(
            [_personal_structured_features(item) for item in items]
        ).tocsr()
        parts.append(structured)
        block_rows["structured"] = structured
    if space.lexical is not None:
        lexical = space.lexical.transform(
            [_model_text(item) for item in items]
        ).tocsr()
        parts.append(lexical)
        block_rows["lexical"] = lexical
    if "dense" in _blocks(space.variant):
        if dense_embeddings is None or len(dense_embeddings) != len(items):
            raise ValueError(
                "Les vecteurs sémantiques ne correspondent pas aux films."
            )
        dense = csr_matrix(np.asarray(dense_embeddings, dtype=np.float64))
        parts.append(dense)
        block_rows["dense"] = dense
    if not parts:
        raise ValueError("La variante ne contient aucun bloc exploitable.")
    return (
        parts[0] if len(parts) == 1 else hstack(parts, format="csr"),
        block_rows,
    )


def _benchmarks(items: list[dict], fallback: float) -> np.ndarray:
    return np.asarray(
        [pm._benchmark(item, fallback) for item in items],
        dtype=float,
    )


def _ranking_metrics(
    scores: np.ndarray,
    ratings: np.ndarray,
) -> dict[str, float | None]:
    scores = np.asarray(scores, dtype=float)
    ratings = np.asarray(ratings, dtype=float)
    labels = (ratings >= pm.LIKE_THRESHOLD).astype(int)
    result: dict[str, float | None] = {
        "auc": (
            float(roc_auc_score(labels, scores))
            if len(np.unique(labels)) >= 2
            else None
        )
    }
    for size in (10, 20, 50):
        k = max(1, min(size, len(scores)))
        indexes = np.argsort(scores)[::-1][:k]
        result[f"precision_at_{size}"] = float(np.mean(labels[indexes]))
        result[f"ndcg_at_{size}"] = float(
            ndcg_score([labels], [scores], k=k)
        )
        result[f"average_rating_at_{size}"] = float(
            np.mean(ratings[indexes])
        )
    return result


def _selection_key(row: dict) -> tuple[float, float, float, float]:
    return (
        float(row.get("ndcg_at_10") or 0.0),
        float(row.get("precision_at_10") or 0.0),
        float(row.get("ndcg_at_20") or 0.0),
        -float(row.get("mae") or 99.0),
    )


def _expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 5,
) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if not len(probabilities):
        return 0.0
    order = np.argsort(probabilities)
    groups = np.array_split(order, min(bins, len(order)))
    return float(
        sum(
            len(indexes)
            / len(probabilities)
            * abs(
                float(np.mean(probabilities[indexes]))
                - float(np.mean(labels[indexes]))
            )
            for indexes in groups
            if len(indexes)
        )
    )


def _oof_predictions(
    items: list[dict],
    dense_embeddings: np.ndarray | None,
    variant: str,
    *,
    alphas: tuple[float, ...] = HYBRID_ALPHAS,
    forced_alpha: float | None = None,
) -> tuple[float, np.ndarray, list[dict]]:
    if forced_alpha is not None:
        alphas = (float(forced_alpha),)
    ratings = np.asarray([float(item["rating"]) for item in items], dtype=float)
    labels = (ratings >= pm.LIKE_THRESHOLD).astype(int)
    folds = min(4, int(labels.sum()), int(len(labels) - labels.sum()))
    if folds < 3:
        raise ValueError("Pas assez de notes contrastées pour le challenger.")
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=8101979,
    )
    predictions = {
        float(alpha): np.zeros(len(items), dtype=float)
        for alpha in alphas
    }
    fold_metrics: dict[float, list[dict]] = {
        float(alpha): [] for alpha in alphas
    }
    for train_indexes, test_indexes in splitter.split(
        np.zeros(len(items)),
        labels,
    ):
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
        baseline = float(np.mean(ratings[train_indexes]))
        space = _fit_space(train, variant)
        train_matrix, _ = _transform(space, train, train_dense)
        test_matrix, _ = _transform(space, test, test_dense)
        targets = ratings[train_indexes] - baseline
        for alpha in alphas:
            value = float(alpha)
            regressor = Ridge(alpha=value, solver="lsqr")
            regressor.fit(train_matrix, targets)
            predicted = np.clip(
                baseline + regressor.predict(test_matrix),
                1.0,
                10.0,
            )
            predictions[value][test_indexes] = predicted
            fold_metrics[value].append(
                {
                    **_ranking_metrics(
                        predicted,
                        ratings[test_indexes],
                    ),
                    "mae": float(
                        mean_absolute_error(
                            ratings[test_indexes],
                            predicted,
                        )
                    ),
                }
            )
    selection_rows: list[dict] = []
    for alpha, values in predictions.items():
        rows = fold_metrics[alpha]
        selection_rows.append(
            {
                "alpha": alpha,
                "ndcg_at_10": float(
                    np.mean([row["ndcg_at_10"] for row in rows])
                ),
                "precision_at_10": float(
                    np.mean([row["precision_at_10"] for row in rows])
                ),
                "ndcg_at_20": float(
                    np.mean([row["ndcg_at_20"] for row in rows])
                ),
                "mae": float(mean_absolute_error(ratings, values)),
            }
        )
    selected = (
        next(
            row
            for row in selection_rows
            if float(row["alpha"]) == float(forced_alpha)
        )
        if forced_alpha is not None
        else max(selection_rows, key=_selection_key)
    )
    alpha = float(selected["alpha"])
    return alpha, predictions[alpha], selection_rows


def _fit_calibrator(
    predicted_ratings: np.ndarray,
    ratings: np.ndarray,
) -> LogisticRegression:
    labels = (ratings >= pm.LIKE_THRESHOLD).astype(int)
    calibrator = LogisticRegression(C=12.0, solver="lbfgs")
    calibrator.fit(
        np.asarray(predicted_ratings, dtype=float).reshape(-1, 1),
        labels,
    )
    return calibrator


def fit_hybrid_model(
    items: list[dict],
    dense_embeddings: np.ndarray | None,
    *,
    variant: str,
    fingerprint: str,
    fixed_alpha: float | None = None,
) -> HybridTasteModel:
    ratings = np.asarray([float(item["rating"]) for item in items], dtype=float)
    baseline = float(np.mean(ratings))
    targets = ratings - baseline
    alpha, oof_predictions, selection_rows = _oof_predictions(
        items,
        dense_embeddings,
        variant,
        forced_alpha=fixed_alpha,
    )
    calibrator = _fit_calibrator(oof_predictions, ratings)
    raw_probabilities = calibrator.predict_proba(
        oof_predictions.reshape(-1, 1)
    )[:, 1]
    labels = (ratings >= pm.LIKE_THRESHOLD).astype(int)
    space = _fit_space(items, variant)
    matrix, _ = _transform(space, items, dense_embeddings)
    regressor = Ridge(alpha=alpha, solver="lsqr")
    regressor.fit(matrix, targets)
    errors = np.abs(ratings - oof_predictions)
    metrics = {
        **_ranking_metrics(oof_predictions, ratings),
        "mae": float(mean_absolute_error(ratings, oof_predictions)),
        "brier": float(brier_score_loss(labels, raw_probabilities)),
        "calibration_bias": float(
            np.mean(raw_probabilities) - np.mean(labels)
        ),
        "ece_global": _expected_calibration_error(
            raw_probabilities,
            labels,
        ),
        "base_like_rate": float(np.mean(labels)),
        "selected_alpha": alpha,
        "alpha_search": selection_rows,
        "variant": variant,
        "variant_label": HYBRID_VARIANTS[variant]["label"],
    }
    return HybridTasteModel(
        version=HYBRID_MODEL_VERSION,
        variant=variant,
        fingerprint=fingerprint,
        trained_at=datetime.now(UTC).isoformat(),
        rated_count=len(items),
        selected_alpha=alpha,
        space=space,
        regressor=regressor,
        calibrator=calibrator,
        user_baseline=baseline,
        mean_personal_residual=0.0,
        base_like_rate=float(np.mean(labels)),
        error_p80=float(np.quantile(errors, 0.80)),
        metrics=metrics,
    )


def _block_coverage(
    model: HybridTasteModel,
    item: dict,
    blocks: dict[str, csr_matrix | None],
) -> float:
    coverages: list[float] = []
    structured = blocks.get("structured")
    if structured is not None and model.space.structured is not None:
        categorical = [
            name
            for name in _personal_structured_features(item)
            if not name.startswith("num::")
        ]
        known = sum(
            name in model.space.structured.vocabulary_
            for name in categorical
        )
        coverages.append(
            min(1.0, 0.35 + 0.65 * known / max(1, len(categorical)))
        )
    lexical = blocks.get("lexical")
    if lexical is not None:
        coverages.append(
            min(
                1.0,
                0.25
                + 0.75
                * min(1.0, float(lexical.getnnz()) / 18.0),
            )
        )
    if blocks.get("dense") is not None:
        coverages.append(pm._description_coverage(item))
    return float(np.mean(coverages)) if coverages else 0.0


def _feature_names(space: HybridFeatureSpace) -> list[str]:
    names: list[str] = []
    if space.structured is not None:
        names.extend(space.structured.get_feature_names_out())
    if space.lexical is not None:
        names.extend(
            "text::" + value
            for value in space.lexical.get_feature_names_out()
        )
    return [str(name) for name in names]


def _learned_signals(
    model: HybridTasteModel,
    matrix: csr_matrix,
) -> tuple[list[dict], list[dict]]:
    names = _feature_names(model.space)
    contributions: list[tuple[float, str]] = []
    row = matrix.getrow(0)
    for index, value in zip(row.indices, row.data, strict=True):
        if index >= len(names) or index >= len(model.regressor.coef_):
            continue
        name = names[int(index)]
        if name.startswith("num::") or name.startswith("public_source::"):
            continue
        contribution = float(value) * float(model.regressor.coef_[int(index)])
        if abs(contribution) < 0.015:
            continue
        label = (
            "Description : " + name.removeprefix("text::")
            if name.startswith("text::")
            else pm._feature_label(name)
        )
        contributions.append((contribution, label))
    positive = sorted(
        [row for row in contributions if row[0] > 0],
        reverse=True,
    )[:5]
    negative = sorted(
        [row for row in contributions if row[0] < 0],
        key=lambda row: row[0],
    )[:5]
    return (
        [
            {"label": label, "impact": round(value, 2)}
            for value, label in positive
        ],
        [
            {"label": label, "impact": round(value, 2)}
            for value, label in negative
        ],
    )


def predict_hybrid_items(
    model: HybridTasteModel,
    items: list[dict],
    dense_embeddings: np.ndarray | None,
) -> list[dict]:
    if not items:
        return []
    matrix, blocks = _transform(model.space, items, dense_embeddings)
    raw_deviations = np.asarray(model.regressor.predict(matrix), dtype=float)
    results: list[dict] = []
    for index, item in enumerate(items):
        row_matrix = matrix.getrow(index)
        row_blocks = {
            name: (
                block.getrow(index) if block is not None else None
            )
            for name, block in blocks.items()
        }
        coverage = _block_coverage(model, item, row_blocks)
        personal_factor = 0.45 + 0.55 * coverage
        deviation = raw_deviations[index] * personal_factor
        predicted = float(
            np.clip(model.user_baseline + deviation, 1.0, 10.0)
        )
        raw_probability = float(
            model.calibrator.predict_proba([[predicted]])[0, 1]
        )
        reliability = float(item.get("benchmark_reliability") or 0.0)
        probability = pm._conservative_probability(
            raw_probability,
            pm._probability_strength(coverage, reliability),
            model.base_like_rate,
        )
        confidence = pm._evidence_confidence(
            float(model.metrics["mae"]),
            coverage,
            reliability,
        )
        half_interval = max(
            0.5,
            model.error_p80 * (1.45 - 0.45 * confidence),
        )
        positive, negative = _learned_signals(model, row_matrix)
        results.append(
            {
                "like_probability": round(100.0 * probability, 1),
                "raw_like_probability": round(
                    100.0 * raw_probability,
                    1,
                ),
                "predicted_rating": round(predicted, 2),
                "prediction_low": round(
                    max(1.0, predicted - half_interval),
                    1,
                ),
                "prediction_high": round(
                    min(10.0, predicted + half_interval),
                    1,
                ),
                "confidence": round(100.0 * confidence, 1),
                "coverage": round(100.0 * coverage, 1),
                "public_baseline_rating": round(
                    float(pm._benchmark(item, model.user_baseline)),
                    2,
                ),
                "user_baseline_rating": round(model.user_baseline, 2),
                "public_raw_rating": item.get("benchmark_raw"),
                "public_rating_reliability": round(
                    100.0 * reliability,
                    1,
                ),
                "personal_adjustment": round(
                    predicted - model.user_baseline,
                    2,
                ),
                "public_influence_weight": 0.0,
                "positive_signals": positive,
                "negative_signals": negative,
                "model_version": model.version,
                "engine": "personal_v09",
                "variant": model.variant,
                "variant_label": HYBRID_VARIANTS[model.variant]["label"],
                "positive_island": None,
                "negative_island": None,
            }
        )
    return results


def evaluate_hybrid_split(
    train: list[dict],
    test: list[dict],
    train_dense: np.ndarray | None,
    test_dense: np.ndarray | None,
    *,
    variant: str,
    split_name: str,
) -> tuple[dict, dict]:
    fingerprint = (
        f"audit-hybrid:{variant}:{split_name}:{len(train)}:{len(test)}"
    )
    model = fit_hybrid_model(
        train,
        train_dense,
        variant=variant,
        fingerprint=fingerprint,
    )
    predictions = predict_hybrid_items(model, test, test_dense)
    scores = np.asarray(
        [float(row["like_probability"]) / 100.0 for row in predictions],
        dtype=float,
    )
    predicted_ratings = np.asarray(
        [float(row["predicted_rating"]) for row in predictions],
        dtype=float,
    )
    ratings = np.asarray([float(item["rating"]) for item in test], dtype=float)
    labels = (ratings >= pm.LIKE_THRESHOLD).astype(int)
    metrics = {
        **_ranking_metrics(scores, ratings),
        "mae": float(mean_absolute_error(ratings, predicted_ratings)),
        "brier": float(brier_score_loss(labels, scores)),
        "mean_probability": float(np.mean(scores)),
        "calibration_bias": float(np.mean(scores) - np.mean(labels)),
        "ece_global": _expected_calibration_error(scores, labels),
        "positive_rate": float(np.mean(labels)),
        "mean_confidence": float(
            np.mean([float(row["confidence"]) / 100.0 for row in predictions])
        ),
        "mean_coverage": float(
            np.mean([float(row["coverage"]) / 100.0 for row in predictions])
        ),
        "engine": f"personal_{variant}_v09",
        "variant": variant,
        "variant_label": HYBRID_VARIANTS[variant]["label"],
        "split": split_name,
        "selected_alpha": float(model.selected_alpha),
    }
    for size in (10, 20):
        k = max(1, min(size, len(scores)))
        indexes = np.argsort(scores)[::-1][:k]
        metrics[f"ece_at_{size}"] = _expected_calibration_error(
            scores[indexes],
            labels[indexes],
        )
    detail = {
        "variant": variant,
        "selected_alpha": float(model.selected_alpha),
        "alpha_search": model.metrics["alpha_search"],
    }
    if split_name.startswith("chronological"):
        top = np.argsort(scores)[::-1][:20]
        detail["top_20_examples"] = [
            {
                "title": test[int(index)]["title"],
                "date_rated": test[int(index)].get("date_rated"),
                "actual_rating": float(test[int(index)]["rating"]),
                "score": round(float(scores[int(index)]), 4),
            }
            for index in top
        ]
    return metrics, detail


def active_configuration(
    database: str | Path | None,
) -> dict | None:
    initialize(database)
    with connect(database) as connection:
        row = connection.execute(
            """
            SELECT engine, configuration_json, audit_created_at, applied_at
            FROM active_model_configuration
            WHERE singleton_id=1
            """
        ).fetchone()
    if not row:
        return {
            "engine": "personal_v09",
            "configuration": {
                "variant": DEFAULT_PERSONAL_VARIANT,
                "variant_label": HYBRID_VARIANTS[
                    DEFAULT_PERSONAL_VARIANT
                ]["label"],
                "semantic_model": None,
                "selected_alpha": None,
            },
            "audit_created_at": None,
            "applied_at": None,
            "automatic": True,
        }
    try:
        configuration = json.loads(row["configuration_json"])
    except (TypeError, json.JSONDecodeError):
        return None
    stored_engine = str(row["engine"])
    return {
        "engine": (
            "personal_v09"
            if stored_engine == "hybrid_v08"
            else stored_engine
        ),
        "configuration": configuration,
        "audit_created_at": row["audit_created_at"],
        "applied_at": row["applied_at"],
    }


def apply_configuration(
    database: str | Path | None,
    *,
    variant: str,
    audit_created_at: str | None,
    selected_alpha: float | None = None,
) -> None:
    _blocks(variant)
    payload = {
        "variant": variant,
        "variant_label": HYBRID_VARIANTS[variant]["label"],
        "semantic_model": (
            SEMANTIC_MODEL if "dense" in _blocks(variant) else None
        ),
        "selected_alpha": (
            float(selected_alpha) if selected_alpha is not None else None
        ),
    }
    with transaction(database) as connection:
        connection.execute(
            """
            INSERT INTO active_model_configuration(
              singleton_id, engine, configuration_json,
              audit_created_at, applied_at
            ) VALUES (1, 'personal_v09', ?, ?, ?)
            ON CONFLICT(singleton_id) DO UPDATE SET
              engine=excluded.engine,
              configuration_json=excluded.configuration_json,
              audit_created_at=excluded.audit_created_at,
              applied_at=excluded.applied_at
            """,
            (
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                audit_created_at,
                datetime.now(UTC).isoformat(),
            ),
        )


def restore_linear_configuration(
    database: str | Path | None,
) -> None:
    with transaction(database) as connection:
        connection.execute(
            """
            INSERT INTO active_model_configuration(
              singleton_id, engine, configuration_json,
              audit_created_at, applied_at
            ) VALUES (1, 'linear_v06', '{}', NULL, ?)
            ON CONFLICT(singleton_id) DO UPDATE SET
              engine=excluded.engine,
              configuration_json=excluded.configuration_json,
              audit_created_at=NULL,
              applied_at=excluded.applied_at
            """,
            (datetime.now(UTC).isoformat(),),
        )


def _cache_version(variant: str, selected_alpha: float | None) -> str:
    alpha_label = (
        "auto"
        if selected_alpha is None
        else f"{float(selected_alpha):g}"
    )
    return f"{HYBRID_MODEL_VERSION}:{variant}:alpha={alpha_label}"


def ensure_hybrid_model(
    database: str | Path | None,
    *,
    force: bool = False,
) -> HybridModelState:
    configuration = active_configuration(database)
    if not configuration or configuration["engine"] not in {
        "hybrid_v08",
        "personal_v09",
    }:
        return HybridModelState(
            "inactive",
            {"message": "La v0.6 linéaire reste active."},
        )
    variant = str(configuration["configuration"].get("variant") or "")
    _blocks(variant)
    configured_alpha = configuration["configuration"].get("selected_alpha")
    selected_alpha = (
        float(configured_alpha) if configured_alpha is not None else None
    )
    items = pm._load_training_items(database)
    if len(items) < pm.MINIMUM_RATINGS:
        return HybridModelState(
            "insufficient",
            {"message": "Historique insuffisant pour le moteur hybride."},
        )
    fingerprint = pm._fingerprint(items)
    model_version = _cache_version(variant, selected_alpha)
    if not force:
        with connect(database) as connection:
            row = connection.execute(
                """
                SELECT model_blob, metrics_json
                FROM personal_models
                WHERE model_version=? AND fingerprint=?
                ORDER BY id DESC LIMIT 1
                """,
                (model_version, fingerprint),
            ).fetchone()
        if row:
            try:
                model = pickle.loads(row["model_blob"])
                summary = json.loads(row["metrics_json"])
                if isinstance(model, HybridTasteModel):
                    return HybridModelState("ready", summary, model)
            except Exception:
                configure_logging(database).warning(
                    "hybrid_model_cache_invalid | version=%s",
                    model_version,
                    exc_info=True,
                )
    try:
        dense = (
            prepare_dense_embeddings(database, items)
            if "dense" in _blocks(variant)
            else None
        )
        model = fit_hybrid_model(
            items,
            dense,
            variant=variant,
            fingerprint=fingerprint,
            fixed_alpha=selected_alpha,
        )
    except Exception as exc:
        configure_logging(database).exception(
            "hybrid_model_training_failed | variant=%s",
            variant,
        )
        return HybridModelState(
            "error",
            {
                "message": str(exc),
                "variant": variant,
            },
        )
    summary_json = json.dumps(
        model.metrics,
        ensure_ascii=False,
        sort_keys=True,
    )
    with transaction(database) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO personal_models(
              model_version, fingerprint, trained_at, rated_count,
              model_blob, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                model_version,
                fingerprint,
                model.trained_at,
                model.rated_count,
                pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL),
                summary_json,
            ),
        )
    return HybridModelState("ready", model.metrics, model)


def predict_hybrid_candidates(
    database: str | Path | None,
    model: HybridTasteModel,
    candidates: list[dict],
) -> list[dict]:
    items = [pm._candidate_item(candidate) for candidate in candidates]
    dense = (
        prepare_dense_embeddings(
            database,
            items,
            kind="candidate",
        )
        if "dense" in _blocks(model.variant)
        else None
    )
    return predict_hybrid_items(model, items, dense)
