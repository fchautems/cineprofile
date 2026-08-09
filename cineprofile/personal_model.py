from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .db import connect, initialize, transaction
from .diagnostics import configure_logging
from .media_types import is_series_type
from .public_rating import best_public_rating, public_rating
from .semantic import _document_text
from .taste_islands import (
    TASTE_ISLANDS_VERSION,
    TasteIslandsPredictor,
    fit_island_space,
    predict_taste_islands,
    transform_island_features,
)


PERSONAL_MODEL_VERSION = "cineprofile-personal-0.9.0"
MINIMUM_RATINGS = 60
LIKE_THRESHOLD = 8.0
RIDGE_ALPHAS = (6.0, 18.0, 50.0)
ISLAND_ALPHAS = (1.0, 4.0, 12.0)

PERSON_WEIGHTS = {
    "directors": 0.40,
    "writers": 0.25,
    "actors": 0.20,
    "cinematographers": 0.05,
    "composers": 0.05,
    "editors": 0.05,
}
PERSON_MINIMUMS = {
    "directors": 2,
    "writers": 2,
    "actors": 5,
    "cinematographers": 3,
    "composers": 3,
    "editors": 3,
}
PROFILE_SHRINKAGE = {
    "genres": 8.0,
    "keywords": 5.0,
    "directors": 3.0,
    "writers": 3.0,
    "actors": 6.0,
    "cinematographers": 4.0,
    "composers": 4.0,
    "editors": 4.0,
}
LEGACY_WEIGHTS = {
    "genres": 0.10,
    "keywords": 0.20,
    "people": 0.15,
    "semantic": 0.40,
    "explicit": 0.15,
}


@dataclass
class FeatureSpace:
    structured: DictVectorizer
    text: TfidfVectorizer | None


@dataclass
class PersonalTasteModel:
    version: str
    trained_at: str
    fingerprint: str
    rated_count: int
    user_baseline: float
    mean_personal_residual: float
    selected_alpha: float
    space: FeatureSpace
    regressor: Ridge
    calibrator: LogisticRegression
    error_p80: float
    islands: TasteIslandsPredictor | None
    metrics: dict


@dataclass
class PersonalModelState:
    status: str
    summary: dict
    model: PersonalTasteModel | None = None


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normalise_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _json_names(value: object) -> list[str]:
    if not value:
        return []
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    result: list[str] = []
    if not isinstance(payload, list):
        return result
    for item in payload:
        name = item.get("name") if isinstance(item, dict) else None
        if name:
            result.append(str(name))
    return result


def _empty_entities() -> dict[str, list[tuple[str, str]]]:
    return {
        "genres": [],
        "keywords": [],
        "directors": [],
        "writers": [],
        "actors": [],
        "cinematographers": [],
        "composers": [],
        "editors": [],
    }


def _load_training_items(
    database: str | Path | None,
) -> list[dict]:
    initialize(database)
    with connect(database) as connection:
        title_rows = connection.execute(
            """
            SELECT imdb_id, title, title_type, year, user_rating, date_rated,
                   imdb_rating, tmdb_rating, num_votes, tmdb_vote_count,
                   runtime_minutes, release_date, genres_csv, directors_csv,
                   overview, tagline, original_language, countries_json,
                   companies_json, metadata_status, enriched_at
            FROM titles
            ORDER BY imdb_id
            """
        ).fetchall()
        genre_rows = connection.execute(
            """
            SELECT tg.imdb_id, CAST(g.tmdb_id AS TEXT) AS entity_id, g.name
            FROM title_genres tg JOIN genres g ON g.tmdb_id=tg.genre_id
            """
        ).fetchall()
        keyword_rows = connection.execute(
            """
            SELECT tk.imdb_id, CAST(k.tmdb_id AS TEXT) AS entity_id, k.name
            FROM title_keywords tk JOIN keywords k ON k.tmdb_id=tk.keyword_id
            """
        ).fetchall()
        credit_rows = connection.execute(
            """
            SELECT c.imdb_id, CAST(p.tmdb_id AS TEXT) AS entity_id, p.name,
                   c.role, c.credit_order
            FROM credits c JOIN people p ON p.tmdb_id=c.person_id
            ORDER BY c.imdb_id, c.role, c.credit_order, p.name
            """
        ).fetchall()

    source_rows = {row["imdb_id"]: row for row in title_rows}
    items: dict[str, dict] = {}
    for row in title_rows:
        if is_series_type(row["title_type"]):
            continue
        rating_evidence = best_public_rating(
            tmdb_rating=row["tmdb_rating"],
            tmdb_votes=row["tmdb_vote_count"],
            imdb_rating=row["imdb_rating"],
            imdb_votes=row["num_votes"],
        )
        item = {
            "id": row["imdb_id"],
            "title": row["title"],
            "title_type": row["title_type"],
            "year": _number(row["year"]),
            "rating": float(row["user_rating"]),
            "date_rated": row["date_rated"],
            "benchmark": rating_evidence.adjusted_rating,
            "benchmark_raw": rating_evidence.raw_rating,
            "benchmark_source": rating_evidence.source,
            "benchmark_reliability": rating_evidence.reliability,
            "votes": rating_evidence.vote_count,
            "runtime": _number(row["runtime_minutes"]),
            "release_date": row["release_date"],
            "overview": row["overview"],
            "tagline": row["tagline"],
            "language": row["original_language"],
            "countries": _json_names(row["countries_json"]),
            "companies": _json_names(row["companies_json"]),
            "metadata_status": row["metadata_status"],
            "enriched_at": row["enriched_at"],
            "entities": _empty_entities(),
        }
        items[row["imdb_id"]] = item

    for row in genre_rows:
        if row["imdb_id"] in items:
            items[row["imdb_id"]]["entities"]["genres"].append(
                (row["entity_id"], row["name"])
            )
    for row in keyword_rows:
        if row["imdb_id"] in items:
            items[row["imdb_id"]]["entities"]["keywords"].append(
                (row["entity_id"], row["name"])
            )
    for row in credit_rows:
        item = items.get(row["imdb_id"])
        if not item:
            continue
        role = str(row["role"])
        if role == "cast":
            if row["credit_order"] is not None and int(row["credit_order"]) >= 5:
                continue
            dimension = "actors"
        else:
            dimension = {
                "director": "directors",
                "writer": "writers",
                "cinematography": "cinematographers",
                "composer": "composers",
                "editor": "editors",
            }.get(role)
        if dimension:
            item["entities"][dimension].append(
                (row["entity_id"], row["name"])
            )

    for item in items.values():
        source = source_rows[item["id"]]
        if not item["entities"]["genres"]:
            for name in str(source["genres_csv"] or "").split(","):
                name = name.strip()
                if name:
                    item["entities"]["genres"].append(
                        ("csv:" + _normalise_name(name), name)
                    )
        if not item["entities"]["directors"]:
            for name in str(source["directors_csv"] or "").split(","):
                name = name.strip()
                if name:
                    item["entities"]["directors"].append(
                        ("csv:" + _normalise_name(name), name)
                    )
    return list(items.values())


def _category_feature(prefix: str, name: object) -> str:
    return f"{prefix}::{_normalise_name(name)}"


def _structured_features(item: dict) -> dict[str, float]:
    features: dict[str, float] = {"num::bias": 1.0}
    benchmark = _number(item.get("benchmark"))
    votes = _number(item.get("votes"))
    runtime = _number(item.get("runtime"))
    features["num::public_rating"] = (
        (benchmark - 7.0) / 2.0 if benchmark is not None else 0.0
    )
    features["num::public_missing"] = 1.0 if benchmark is None else 0.0
    features["num::public_reliability"] = float(
        item.get("benchmark_reliability") or 0.0
    )
    features["num::log_votes"] = (
        math.log1p(max(0.0, votes)) / 14.0 if votes is not None else 0.0
    )
    features["num::runtime"] = (
        (runtime - 110.0) / 80.0 if runtime is not None else 0.0
    )
    if runtime is not None:
        band = (
            "moins de 90"
            if runtime < 90
            else "90 à 119"
            if runtime < 120
            else "120 à 149"
            if runtime < 150
            else "150 et plus"
        )
        features[_category_feature("runtime", band)] = 1.0
    if item.get("language"):
        features[_category_feature("language", item["language"])] = 1.0
    if item.get("benchmark_source"):
        features[
            _category_feature(
                "public_source",
                item["benchmark_source"],
            )
        ] = 1.0
    if item.get("title_type"):
        features[_category_feature("type", item["title_type"])] = 1.0
    for country in item.get("countries", []):
        features[_category_feature("country", country)] = 1.0
    for company in item.get("companies", [])[:8]:
        features[_category_feature("company", company)] = 1.0
    for dimension, entities in item.get("entities", {}).items():
        for _, name in entities:
            features[_category_feature(dimension, name)] = 1.0
    return features


def _item_text(item: dict) -> str:
    entities = item.get("entities", {})
    return _document_text(
        {
            "overview": item.get("overview"),
            "tagline": item.get("tagline"),
            "genres": [name for _, name in entities.get("genres", [])],
            "keywords": [name for _, name in entities.get("keywords", [])],
        }
    )


def _fit_space(items: list[dict]) -> FeatureSpace:
    structured = DictVectorizer(sparse=True, dtype=np.float64)
    structured.fit([_structured_features(item) for item in items])
    text: TfidfVectorizer | None = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=7000,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    try:
        text.fit([_item_text(item) for item in items])
    except ValueError:
        text = None
    return FeatureSpace(structured=structured, text=text)


def _transform(
    space: FeatureSpace,
    items: list[dict],
) -> tuple[csr_matrix, csr_matrix | None]:
    structured = space.structured.transform(
        [_structured_features(item) for item in items]
    ).tocsr()
    text = (
        space.text.transform([_item_text(item) for item in items]).tocsr()
        if space.text is not None
        else None
    )
    matrix = hstack([structured, text], format="csr") if text is not None else structured
    return matrix, text


def _description_coverage(item: dict) -> float:
    overview = str(item.get("overview") or "").strip()
    tagline = str(item.get("tagline") or "").strip()
    if len(overview) >= 120:
        return 1.0
    if len(overview) >= 50:
        return 0.75
    if overview:
        return 0.45
    if tagline:
        return 0.25
    return 0.05


def _space_coverage(
    space: FeatureSpace,
    item: dict,
    text_row: csr_matrix | None,
) -> float:
    features = _structured_features(item)
    categorical = [
        name for name in features if not name.startswith("num::")
    ]
    known = sum(
        1 for name in categorical if name in space.structured.vocabulary_
    )
    structured = known / len(categorical) if categorical else 0.15
    text = (
        min(1.0, float(text_row.nnz) / 14.0)
        if text_row is not None
        else 0.0
    )
    description = _description_coverage(item)
    return max(
        0.05,
        min(
            1.0,
            0.55 * structured + 0.25 * text + 0.20 * description,
        ),
    )


def _model_reliability(mae: float) -> float:
    return max(0.35, min(0.92, 1.0 - float(mae) / 3.2))


def _evidence_confidence(
    mae: float,
    coverage: float,
    public_reliability: float,
    *,
    separation: float = 1.0,
) -> float:
    return max(
        0.05,
        min(
            0.95,
            _model_reliability(mae)
            * (0.45 + 0.55 * max(0.0, min(1.0, coverage)))
            * (
                0.60
                + 0.40
                * max(0.0, min(1.0, public_reliability))
            )
            * max(0.55, min(1.0, separation)),
        ),
    )


def _conservative_probability(
    raw_probability: float,
    confidence: float,
    base_rate: float,
) -> float:
    return max(
        0.01,
        min(
            0.99,
            base_rate
            + confidence * (float(raw_probability) - base_rate),
        ),
    )


def _probability_strength(
    coverage: float,
    public_reliability: float,
    *,
    separation: float = 1.0,
) -> float:
    metadata_factor = max(0.0, min(1.0, coverage))
    public_factor = (
        0.75 + 0.25 * max(0.0, min(1.0, public_reliability))
    )
    separation_factor = (
        0.85 + 0.15 * max(0.55, min(1.0, separation))
    )
    evidence = math.sqrt(
        metadata_factor * public_factor * separation_factor
    )
    return max(0.50, min(1.0, 0.50 + 0.50 * evidence))


def _benchmark(item: dict, fallback: float) -> float:
    return float(_number(item.get("benchmark")) or fallback)


def _dimension_evidence(rows: Iterable[dict]) -> tuple[float, float] | None:
    observations = list(rows)
    if not observations:
        return None
    weights = np.array(
        [max(0.05, float(row["confidence"])) for row in observations],
        dtype=float,
    )
    signals = np.array([float(row["affinity"]) for row in observations])
    signal = float(np.average(signals, weights=weights))
    score = 0.5 + 0.5 * math.tanh(signal)
    confidence = min(
        1.0,
        float(np.mean([row["confidence"] for row in observations]))
        * (1.0 + 0.10 * math.log1p(len(observations))),
    )
    return score, confidence


def _weighted_available(
    values: dict[str, tuple[float, float] | None],
    weights: dict[str, float],
) -> tuple[float, float]:
    available = {key: value for key, value in values.items() if value is not None}
    if not available:
        return 0.5, 0.0
    available_weight = sum(weights[key] for key in available)
    score = sum(
        weights[key] * float(value[0]) for key, value in available.items()
    ) / available_weight
    confidence = sum(
        weights[key] * float(value[1]) for key, value in available.items()
    ) / sum(weights.values())
    return float(score), min(1.0, float(confidence))


def _entity_statistics(
    train: list[dict],
    baseline: float,
) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for dimension in PROFILE_SHRINKAGE:
        buckets: dict[str, list[tuple[float, float | None]]] = {}
        for item in train:
            for entity_id, _ in item["entities"].get(dimension, []):
                buckets.setdefault(str(entity_id), []).append(
                    (float(item["rating"]), _number(item.get("benchmark")))
                )
        minimum = PERSON_MINIMUMS.get(dimension, 2)
        rows: dict[str, dict] = {}
        for entity_id, observations in buckets.items():
            if len(observations) < minimum:
                continue
            ratings = [row[0] for row in observations]
            residuals = [
                row[0] - row[1] for row in observations if row[1] is not None
            ]
            support = len(observations)
            confidence = support / (
                support + PROFILE_SHRINKAGE[dimension]
            )
            raw = float(np.mean(ratings)) - baseline
            if residuals:
                raw += 0.45 * float(np.mean(residuals))
            rows[entity_id] = {
                "affinity": confidence * raw,
                "confidence": confidence,
            }
        result[dimension] = rows
    return result


def _legacy_semantic(
    train: list[dict],
    test: list[dict],
    train_text: csr_matrix | None,
    test_text: csr_matrix | None,
    baseline: float,
) -> list[tuple[float, float] | None]:
    if train_text is None or test_text is None:
        return [None] * len(test)
    similarities = (test_text @ train_text.T).toarray()
    residuals = np.array(
        [float(item["rating"]) - baseline for item in train],
        dtype=float,
    )
    result: list[tuple[float, float] | None] = []
    for row in similarities:
        selected = [
            int(index)
            for index in np.argsort(row)[::-1][:24]
            if float(row[index]) >= 0.04
        ][:16]
        if not selected:
            result.append(None)
            continue
        values = residuals[selected]
        similarity_values = row[selected]
        weights = np.square(np.maximum(0.03, similarity_values - 0.04 + 0.08))
        signal = float(np.average(values, weights=weights))
        score = 0.5 + 0.5 * math.tanh(signal / 1.5)
        agreement = 1.0 / (1.0 + float(np.std(values)) / 2.0)
        confidence = min(1.0, len(selected) / 8.0) * agreement
        confidence *= min(
            0.55,
            max(0.10, float(np.mean(similarity_values)) / 0.22),
        )
        result.append((score, confidence))
    return result


def _legacy_scores(
    train: list[dict],
    test: list[dict],
    train_text: csr_matrix | None,
    test_text: csr_matrix | None,
) -> np.ndarray:
    baseline = float(np.mean([item["rating"] for item in train]))
    maps = _entity_statistics(train, baseline)
    semantics = _legacy_semantic(
        train,
        test,
        train_text,
        test_text,
        baseline,
    )
    scores: list[float] = []
    for item, semantic in zip(test, semantics, strict=True):
        evidence: dict[str, tuple[float, float] | None] = {}
        for dimension in PROFILE_SHRINKAGE:
            rows = [
                maps[dimension][str(entity_id)]
                for entity_id, _ in item["entities"].get(dimension, [])
                if str(entity_id) in maps[dimension]
            ]
            evidence[dimension] = _dimension_evidence(rows)
        people_score, people_confidence = _weighted_available(
            {key: evidence[key] for key in PERSON_WEIGHTS},
            PERSON_WEIGHTS,
        )
        score, confidence = _weighted_available(
            {
                "genres": evidence["genres"],
                "keywords": evidence["keywords"],
                "people": (
                    (people_score, people_confidence)
                    if people_confidence > 0
                    else None
                ),
                "semantic": semantic,
                "explicit": None,
            },
            LEGACY_WEIGHTS,
        )
        scores.append(0.5 + (score - 0.5) * (0.25 + 0.75 * confidence))
    return np.asarray(scores, dtype=float)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _precision_at(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    size = max(1, min(int(k), len(scores)))
    selected = np.argsort(scores)[::-1][:size]
    return float(np.mean(labels[selected]))


def _ranking_metrics(
    scores: np.ndarray,
    ratings: np.ndarray,
) -> dict[str, float | None]:
    labels = (ratings >= LIKE_THRESHOLD).astype(int)
    k = min(20, len(scores))
    return {
        "auc": _safe_auc(labels, scores),
        "precision_at_20": _precision_at(scores, labels, k),
        "ndcg_at_50": float(
            ndcg_score(
                [labels],
                [scores],
                k=min(50, len(scores)),
            )
        ),
    }


def _probability_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    labels = np.asarray(labels, dtype=int)
    bins: list[dict[str, float | int | str]] = []
    weighted_gap = 0.0
    boundaries = np.linspace(0.0, 1.0, 6)
    for index, (lower, upper) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        selected = (
            (probabilities >= lower) & (probabilities <= upper)
            if index == len(boundaries) - 2
            else (probabilities >= lower) & (probabilities < upper)
        )
        count = int(selected.sum())
        if not count:
            continue
        predicted = float(np.mean(probabilities[selected]))
        observed = float(np.mean(labels[selected]))
        gap = abs(predicted - observed)
        weighted_gap += count * gap
        bins.append(
            {
                "range": f"{int(lower * 100)}–{int(upper * 100)} %",
                "count": count,
                "predicted_rate": round(predicted, 4),
                "observed_rate": round(observed, 4),
                "gap": round(gap, 4),
            }
        )
    return {
        "brier": float(brier_score_loss(labels, probabilities)),
        "calibration_error": (
            float(weighted_gap / len(labels)) if len(labels) else None
        ),
        "calibration_bins": bins,
    }


def _fit_calibrator(
    predicted_ratings: np.ndarray,
    labels: np.ndarray,
    probability_strengths: np.ndarray,
    *,
    folds: int,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    features = np.asarray(predicted_ratings, dtype=float).reshape(-1, 1)
    calibration_splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=21071979,
    )
    prototype = LogisticRegression(C=20.0, solver="lbfgs")
    raw_probabilities = cross_val_predict(
        prototype,
        features,
        labels,
        cv=calibration_splitter,
        method="predict_proba",
    )[:, 1]
    base_rate = float(np.mean(labels))
    probabilities = np.array(
        [
            _conservative_probability(raw, strength, base_rate)
            for raw, strength in zip(
                raw_probabilities,
                probability_strengths,
                strict=True,
            )
        ],
        dtype=float,
    )
    calibrator = LogisticRegression(C=20.0, solver="lbfgs")
    calibrator.fit(features, labels)
    return calibrator, raw_probabilities, probabilities


def _temporal_evaluation(
    items: list[dict],
    alpha: float,
) -> dict | None:
    dated = [
        item
        for item in items
        if item.get("date_rated")
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(item["date_rated"]))
    ]
    if len(dated) < max(MINIMUM_RATINGS, int(0.70 * len(items))):
        return None
    dated.sort(key=lambda item: (item["date_rated"], item["id"]))
    cut = max(MINIMUM_RATINGS, int(0.80 * len(dated)))
    if cut >= len(dated) - 10:
        return None
    train, test = dated[:cut], dated[cut:]
    baseline = float(np.mean([item["rating"] for item in train]))
    space = _fit_space(train)
    train_matrix, _ = _transform(space, train)
    test_matrix, _ = _transform(space, test)
    train_targets = np.array(
        [
            item["rating"] - _benchmark(item, baseline)
            for item in train
        ],
        dtype=float,
    )
    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(train_matrix, train_targets)
    predicted = np.array(
        [_benchmark(item, baseline) for item in test],
        dtype=float,
    ) + model.predict(test_matrix)
    ratings = np.array([item["rating"] for item in test], dtype=float)
    metrics = _ranking_metrics(predicted, ratings)
    metrics["mae"] = float(mean_absolute_error(ratings, predicted))
    metrics["tested_titles"] = len(test)
    return metrics


def _temporal_island_evaluation(
    items: list[dict],
    alpha: float,
) -> dict | None:
    dated = [
        item
        for item in items
        if item.get("date_rated")
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(item["date_rated"]))
    ]
    if len(dated) < max(MINIMUM_RATINGS, int(0.70 * len(items))):
        return None
    dated.sort(key=lambda item: (item["date_rated"], item["id"]))
    cut = max(MINIMUM_RATINGS, int(0.80 * len(dated)))
    if cut >= len(dated) - 10:
        return None
    train, test = dated[:cut], dated[cut:]
    baseline = float(np.mean([item["rating"] for item in train]))
    train_benchmarks = np.array(
        [_benchmark(item, baseline) for item in train],
        dtype=float,
    )
    test_benchmarks = np.array(
        [_benchmark(item, baseline) for item in test],
        dtype=float,
    )
    try:
        space = fit_island_space(train)
    except ValueError:
        return None
    train_features, _ = transform_island_features(space, train)
    test_features, _ = transform_island_features(space, test)
    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(
        train_features,
        np.array([item["rating"] for item in train], dtype=float)
        - train_benchmarks,
    )
    predicted = np.clip(
        test_benchmarks + model.predict(test_features),
        1.0,
        10.0,
    )
    ratings = np.array([item["rating"] for item in test], dtype=float)
    metrics = _ranking_metrics(predicted, ratings)
    metrics["mae"] = float(mean_absolute_error(ratings, predicted))
    metrics["tested_titles"] = len(test)
    return metrics


def _metric_gain(
    challenger: dict,
    champion: dict,
    key: str,
) -> float:
    challenger_value = challenger.get(key)
    champion_value = champion.get(key)
    if challenger_value is None or champion_value is None:
        return 0.0
    return float(challenger_value) - float(champion_value)


def _challenger_wins(
    challenger: dict | None,
    champion: dict,
    legacy: dict,
    challenger_temporal: dict | None,
    champion_temporal: dict | None,
) -> tuple[bool, str]:
    if challenger is None:
        return False, "Le challenger n’a pas pu construire des îlots assez stables."
    auc_gain = _metric_gain(challenger, champion, "auc")
    ndcg_gain = _metric_gain(challenger, champion, "ndcg_at_50")
    mae_gain = float(champion["mae"]) - float(challenger["mae"])
    brier_gain = float(champion["brier"]) - float(challenger["brier"])
    no_regression = (
        auc_gain >= -0.005
        and ndcg_gain >= -0.005
        and mae_gain >= -0.05
        and brier_gain >= -0.005
    )
    clear_gains = sum(
        (
            auc_gain >= 0.01,
            ndcg_gain >= 0.01,
            mae_gain >= 0.08,
            brier_gain >= 0.01,
        )
    )
    legacy_guard = (
        (
            challenger.get("auc") is None
            or legacy.get("auc") is None
            or float(challenger["auc"]) >= float(legacy["auc"]) - 0.005
        )
        and float(challenger["ndcg_at_50"])
        >= float(legacy["ndcg_at_50"]) - 0.005
    )
    temporal_guard = True
    if challenger_temporal and champion_temporal:
        temporal_guard = (
            _metric_gain(
                challenger_temporal,
                champion_temporal,
                "auc",
            )
            >= -0.02
            and float(challenger_temporal["mae"])
            <= float(champion_temporal["mae"]) + 0.10
        )
    if no_regression and clear_gains >= 2 and legacy_guard and temporal_guard:
        return (
            True,
            "La v0.7 gagne clairement sur au moins deux mesures, sans "
            "régression significative ni sur l’historique complet ni sur les "
            "notes les plus récentes.",
        )
    return (
        False,
        "La v0.7 ne gagne pas encore assez nettement sur plusieurs mesures ; "
        "la v0.6 reste le moteur actif.",
    )


def _fingerprint(items: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda row: row["id"]):
        digest.update(
            json.dumps(
                {
                    "id": item["id"],
                    "rating": item["rating"],
                    "benchmark": item["benchmark"],
                    "benchmark_raw": item["benchmark_raw"],
                    "benchmark_source": item["benchmark_source"],
                    "benchmark_reliability": item[
                        "benchmark_reliability"
                    ],
                    "votes": item["votes"],
                    "year": item["year"],
                    "runtime": item["runtime"],
                    "language": item["language"],
                    "countries": item["countries"],
                    "companies": item["companies"],
                    "entities": item["entities"],
                    "text": _item_text(item),
                    "enriched_at": item["enriched_at"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _train_model(items: list[dict], fingerprint: str) -> PersonalTasteModel:
    ratings = np.array([item["rating"] for item in items], dtype=float)
    labels = (ratings >= LIKE_THRESHOLD).astype(int)
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    folds = min(5, positive_count, negative_count)
    if folds < 3:
        raise ValueError(
            "Les notes existantes ne contiennent pas encore assez d’exemples "
            "contrastés pour valider un modèle personnel."
        )

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=26071979,
    )
    predictions = {
        alpha: np.zeros(len(items), dtype=float) for alpha in RIDGE_ALPHAS
    }
    island_predictions = {
        alpha: np.zeros(len(items), dtype=float) for alpha in ISLAND_ALPHAS
    }
    linear_coverages = np.zeros(len(items), dtype=float)
    island_coverages = np.zeros(len(items), dtype=float)
    island_separations = np.ones(len(items), dtype=float)
    public_reliabilities = np.array(
        [
            float(item.get("benchmark_reliability") or 0.0)
            for item in items
        ],
        dtype=float,
    )
    islands_available = True
    island_failure: str | None = None
    public_baseline = np.zeros(len(items), dtype=float)
    legacy = np.zeros(len(items), dtype=float)

    for train_indexes, test_indexes in splitter.split(np.zeros(len(items)), labels):
        train = [items[int(index)] for index in train_indexes]
        test = [items[int(index)] for index in test_indexes]
        baseline = float(np.mean([item["rating"] for item in train]))
        train_benchmarks = np.array(
            [_benchmark(item, baseline) for item in train],
            dtype=float,
        )
        test_benchmarks = np.array(
            [_benchmark(item, baseline) for item in test],
            dtype=float,
        )
        train_targets = np.array(
            [item["rating"] for item in train],
            dtype=float,
        ) - train_benchmarks
        mean_residual = float(np.mean(train_targets))
        public_baseline[test_indexes] = test_benchmarks + mean_residual

        space = _fit_space(train)
        train_matrix, train_text = _transform(space, train)
        test_matrix, test_text = _transform(space, test)
        for local_index, global_index in enumerate(test_indexes):
            linear_coverages[int(global_index)] = _space_coverage(
                space,
                test[local_index],
                (
                    test_text.getrow(local_index)
                    if test_text is not None
                    else None
                ),
            )
        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha, solver="lsqr")
            model.fit(train_matrix, train_targets)
            predictions[alpha][test_indexes] = (
                test_benchmarks + model.predict(test_matrix)
            )
        legacy[test_indexes] = _legacy_scores(
            train,
            test,
            train_text,
            test_text,
        )
        if islands_available:
            try:
                island_space = fit_island_space(train)
                island_train, _ = transform_island_features(
                    island_space,
                    train,
                )
                island_test, island_evidence = transform_island_features(
                    island_space,
                    test,
                )
            except ValueError as exc:
                islands_available = False
                island_failure = str(exc)
            else:
                for local_index, global_index in enumerate(test_indexes):
                    evidence = island_evidence[local_index]
                    island_coverages[int(global_index)] = float(
                        evidence["coverage"]
                    )
                    island_separations[int(global_index)] = min(
                        1.0,
                        0.55 + abs(float(evidence["margin"])),
                    )
                for alpha in ISLAND_ALPHAS:
                    island_model = Ridge(alpha=alpha, solver="lsqr")
                    island_model.fit(island_train, train_targets)
                    island_predictions[alpha][test_indexes] = (
                        test_benchmarks + island_model.predict(island_test)
                    )

    errors = {
        alpha: float(mean_absolute_error(ratings, values))
        for alpha, values in predictions.items()
    }
    selected_alpha = min(errors, key=errors.get)
    selected_predictions = np.clip(
        predictions[selected_alpha],
        1.0,
        10.0,
    )
    linear_mae = float(
        mean_absolute_error(ratings, selected_predictions)
    )
    linear_probability_strengths = np.array(
        [
            _probability_strength(
                coverage,
                public_reliability,
            )
            for coverage, public_reliability in zip(
                linear_coverages,
                public_reliabilities,
                strict=True,
            )
        ],
        dtype=float,
    )
    calibrator, raw_probabilities, probabilities = _fit_calibrator(
        selected_predictions,
        labels,
        linear_probability_strengths,
        folds=folds,
    )
    new_metrics = _ranking_metrics(probabilities, ratings)
    new_metrics["mae"] = linear_mae
    new_metrics.update(_probability_metrics(probabilities, labels))
    new_metrics["raw_brier"] = float(
        brier_score_loss(labels, raw_probabilities)
    )
    legacy_metrics = _ranking_metrics(legacy, ratings)
    public_metrics = _ranking_metrics(public_baseline, ratings)
    public_metrics["mae"] = float(
        mean_absolute_error(ratings, public_baseline)
    )

    island_metrics: dict | None = None
    island_selected_alpha: float | None = None
    island_selected_predictions: np.ndarray | None = None
    island_calibrator: LogisticRegression | None = None
    if islands_available:
        island_errors = {
            alpha: float(mean_absolute_error(ratings, values))
            for alpha, values in island_predictions.items()
        }
        island_selected_alpha = float(min(island_errors, key=island_errors.get))
        island_selected_predictions = np.clip(
            island_predictions[island_selected_alpha],
            1.0,
            10.0,
        )
        island_mae = float(
            mean_absolute_error(ratings, island_selected_predictions)
        )
        island_probability_strengths = np.array(
            [
                _probability_strength(
                    coverage,
                    public_reliability,
                    separation=separation,
                )
                for coverage, public_reliability, separation in zip(
                    island_coverages,
                    public_reliabilities,
                    island_separations,
                    strict=True,
                )
            ],
            dtype=float,
        )
        (
            island_calibrator,
            island_raw_probabilities,
            island_probabilities,
        ) = _fit_calibrator(
            island_selected_predictions,
            labels,
            island_probability_strengths,
            folds=folds,
        )
        island_metrics = _ranking_metrics(island_probabilities, ratings)
        island_metrics["mae"] = island_mae
        island_metrics.update(
            _probability_metrics(island_probabilities, labels)
        )
        island_metrics["raw_brier"] = float(
            brier_score_loss(labels, island_raw_probabilities)
        )

    new_auc = new_metrics.get("auc")
    legacy_auc = legacy_metrics.get("auc")
    validated = (
        new_auc is None
        or legacy_auc is None
        or float(new_auc) >= float(legacy_auc) - 0.01
    ) and (
        float(new_metrics["ndcg_at_50"])
        >= float(legacy_metrics["ndcg_at_50"]) - 0.01
    )

    user_baseline = float(np.mean(ratings))
    full_benchmarks = np.array(
        [_benchmark(item, user_baseline) for item in items],
        dtype=float,
    )
    full_targets = ratings - full_benchmarks
    space = _fit_space(items)
    full_matrix, _ = _transform(space, items)
    regressor = Ridge(alpha=float(selected_alpha), solver="lsqr")
    regressor.fit(full_matrix, full_targets)
    absolute_errors = np.abs(ratings - selected_predictions)

    island_predictor: TasteIslandsPredictor | None = None
    island_summary: dict | None = None
    if (
        island_metrics is not None
        and island_selected_alpha is not None
        and island_selected_predictions is not None
        and island_calibrator is not None
    ):
        full_island_space = fit_island_space(items)
        full_island_features, _ = transform_island_features(
            full_island_space,
            items,
        )
        full_island_regressor = Ridge(
            alpha=island_selected_alpha,
            solver="lsqr",
        )
        full_island_regressor.fit(full_island_features, full_targets)
        island_predictor = TasteIslandsPredictor(
            version=TASTE_ISLANDS_VERSION,
            selected_alpha=island_selected_alpha,
            user_baseline=user_baseline,
            mean_personal_residual=float(np.mean(full_targets)),
            space=full_island_space,
            regressor=full_island_regressor,
            calibrator=island_calibrator,
            error_p80=float(
                np.quantile(
                    np.abs(ratings - island_selected_predictions),
                    0.80,
                )
            ),
            metrics=island_metrics,
        )
        island_summary = {
            "positive": full_island_space.positive_islands,
            "negative": full_island_space.negative_islands,
        }

    trained_at = datetime.now(UTC).isoformat()
    temporal_v06 = _temporal_evaluation(items, float(selected_alpha))
    temporal_v07 = (
        _temporal_island_evaluation(items, island_selected_alpha)
        if island_selected_alpha is not None
        else None
    )
    challenger_validated, selection_reason = _challenger_wins(
        island_metrics,
        new_metrics,
        legacy_metrics,
        temporal_v07,
        temporal_v06,
    )
    if challenger_validated and island_predictor is not None:
        active_engine = "islands_v07"
        active_engine_label = "v0.7 · Îlots recalibrés"
    elif validated:
        active_engine = "linear_v06"
        active_engine_label = "v0.6 · Linéaire recalibré"
    else:
        active_engine = "legacy_v05"
        active_engine_label = "v0.5 · Indice historique"
        selection_reason += (
            " Le contrôle de la v0.6 face à la v0.5 n’étant pas suffisant, "
            "l’indice historique reste actif."
        )
    metrics = {
        "status": "ready",
        "version": PERSONAL_MODEL_VERSION,
        "trained_at": trained_at,
        "rated_count": len(items),
        "positive_count": positive_count,
        "base_like_rate": float(positive_count / len(items)),
        "like_threshold": LIKE_THRESHOLD,
        "folds": folds,
        "selected_alpha": float(selected_alpha),
        "new_model": new_metrics,
        "linear_v06": new_metrics,
        "islands_v07": island_metrics,
        "legacy_v05": legacy_metrics,
        "public_baseline": public_metrics,
        "temporal": temporal_v06,
        "temporal_v06": temporal_v06,
        "temporal_v07": temporal_v07,
        "validated": bool(validated),
        "challenger_validated": bool(challenger_validated),
        "active_engine": active_engine,
        "active_engine_label": active_engine_label,
        "selection_reason": selection_reason,
        "island_failure": island_failure,
        "islands": island_summary,
        "method": (
            "validation croisée commune : chaque note est prédite séparément "
            "par la v0.6 et la v0.7 sans avoir été vue pendant leur apprentissage ; "
            "la calibration est elle aussi évaluée hors échantillon et les notes "
            "publiques sont placées sur une échelle TMDB fiabilisée"
        ),
    }
    return PersonalTasteModel(
        version=PERSONAL_MODEL_VERSION,
        trained_at=trained_at,
        fingerprint=fingerprint,
        rated_count=len(items),
        user_baseline=user_baseline,
        mean_personal_residual=float(np.mean(full_targets)),
        selected_alpha=float(selected_alpha),
        space=space,
        regressor=regressor,
        calibrator=calibrator,
        error_p80=float(np.quantile(absolute_errors, 0.80)),
        islands=island_predictor,
        metrics=metrics,
    )


def _unavailable_summary(items: list[dict]) -> dict:
    return {
        "status": "insufficient",
        "version": PERSONAL_MODEL_VERSION,
        "rated_count": len(items),
        "minimum_ratings": MINIMUM_RATINGS,
        "message": (
            "Le moteur historique reste actif tant qu’il n’y a pas assez de "
            "notes existantes pour effectuer une validation honnête."
        ),
    }


def ensure_personal_model(
    database: str | Path | None = None,
    *,
    force: bool = False,
) -> PersonalModelState:
    initialize(database)
    items = _load_training_items(database)
    if len(items) < MINIMUM_RATINGS:
        summary = _unavailable_summary(items)
        return PersonalModelState("insufficient", summary)
    fingerprint = _fingerprint(items)
    if not force:
        with connect(database) as connection:
            row = connection.execute(
                """
                SELECT model_blob, metrics_json
                FROM personal_models
                WHERE model_version=? AND fingerprint=?
                ORDER BY id DESC LIMIT 1
                """,
                (PERSONAL_MODEL_VERSION, fingerprint),
            ).fetchone()
        if row:
            try:
                model = pickle.loads(row["model_blob"])
                summary = json.loads(row["metrics_json"])
                if isinstance(model, PersonalTasteModel):
                    return PersonalModelState("ready", summary, model)
            except Exception:
                # Un cache de modèle peut devenir illisible après une mise à
                # jour de Python ou d'une dépendance scientifique. Il est
                # entièrement reproductible depuis les notes : on le régénère
                # au lieu de bloquer l'application.
                configure_logging(database).warning(
                    "personal_model_cache_invalid | version=%s | fingerprint=%s",
                    PERSONAL_MODEL_VERSION,
                    fingerprint[:12],
                    exc_info=True,
                )
    try:
        model = _train_model(items, fingerprint)
    except Exception as exc:
        configure_logging(database).exception(
            "personal_model_failed | version=%s | rated_count=%s",
            PERSONAL_MODEL_VERSION,
            len(items),
        )
        summary = {
            "status": "error",
            "version": PERSONAL_MODEL_VERSION,
            "rated_count": len(items),
            "message": str(exc),
        }
        return PersonalModelState("error", summary)
    payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    summary_json = json.dumps(model.metrics, ensure_ascii=False)
    with transaction(database) as connection:
        connection.execute(
            """
            INSERT INTO personal_models(
              model_version, fingerprint, trained_at, rated_count,
              model_blob, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_version, fingerprint) DO UPDATE SET
              trained_at=excluded.trained_at,
              rated_count=excluded.rated_count,
              model_blob=excluded.model_blob,
              metrics_json=excluded.metrics_json
            """,
            (
                PERSONAL_MODEL_VERSION,
                fingerprint,
                model.trained_at,
                model.rated_count,
                payload,
                summary_json,
            ),
        )
    return PersonalModelState("ready", model.metrics, model)


def _candidate_item(candidate: dict) -> dict:
    release_date = str(candidate.get("release_date") or "")
    year = _number(release_date[:4]) if len(release_date) >= 4 else None
    rating_evidence = public_rating(
        candidate.get("vote_average"),
        candidate.get("vote_count"),
        source="tmdb",
    )
    keywords_payload = candidate.get("keywords", {})
    if isinstance(keywords_payload, dict):
        keywords = keywords_payload.get(
            "keywords",
            keywords_payload.get("results", []),
        )
    else:
        keywords = keywords_payload if isinstance(keywords_payload, list) else []
    entities = _empty_entities()
    entities["genres"] = [
        (str(row.get("id", "")), str(row.get("name", "")))
        for row in candidate.get("genres", [])
        if isinstance(row, dict) and row.get("name")
    ]
    entities["keywords"] = [
        (str(row.get("id", "")), str(row.get("name", "")))
        for row in keywords
        if isinstance(row, dict) and row.get("name")
    ]
    for actor in candidate.get("credits", {}).get("cast", [])[:5]:
        if actor.get("name"):
            entities["actors"].append(
                (str(actor.get("id", "")), str(actor["name"]))
            )
    jobs = {
        "Director": "directors",
        "Screenplay": "writers",
        "Writer": "writers",
        "Story": "writers",
        "Director of Photography": "cinematographers",
        "Original Music Composer": "composers",
        "Editor": "editors",
    }
    for person in candidate.get("credits", {}).get("crew", []):
        dimension = jobs.get(str(person.get("job", "")))
        if dimension and person.get("name"):
            entities[dimension].append(
                (str(person.get("id", "")), str(person["name"]))
            )
    return {
        "id": str(candidate.get("id", "")),
        "title": candidate.get("title") or candidate.get("name"),
        "title_type": "movie",
        "year": year,
        "rating": 0.0,
        "date_rated": None,
        "benchmark": rating_evidence.adjusted_rating,
        "benchmark_raw": rating_evidence.raw_rating,
        "benchmark_source": rating_evidence.source,
        "benchmark_reliability": rating_evidence.reliability,
        "votes": rating_evidence.vote_count,
        "runtime": _number(candidate.get("runtime")),
        "release_date": release_date,
        "overview": candidate.get("overview"),
        "tagline": candidate.get("tagline"),
        "language": candidate.get("original_language"),
        "countries": [
            str(row["name"])
            for row in candidate.get("production_countries", [])
            if isinstance(row, dict) and row.get("name")
        ],
        "companies": [
            str(row["name"])
            for row in candidate.get("production_companies", [])
            if isinstance(row, dict) and row.get("name")
        ],
        "entities": entities,
    }


def _coverage(
    model: PersonalTasteModel,
    item: dict,
    text_matrix: csr_matrix | None,
) -> float:
    return _space_coverage(model.space, item, text_matrix)


def _feature_label(name: str) -> str:
    prefix, _, value = name.partition("::")
    labels = {
        "genres": "Genre",
        "keywords": "Thème",
        "directors": "Réalisation",
        "writers": "Scénario",
        "actors": "Interprétation",
        "cinematographers": "Photographie",
        "composers": "Musique",
        "editors": "Montage",
        "language": "Langue",
        "country": "Pays",
        "company": "Production",
        "public_source": "Source publique",
        "runtime": "Durée",
    }
    return f"{labels.get(prefix, prefix.capitalize())} : {value}"


def _learned_signals(
    model: PersonalTasteModel,
    item: dict,
    matrix: csr_matrix,
) -> tuple[list[dict], list[dict]]:
    row = matrix.getrow(0)
    structured_size = len(model.space.structured.vocabulary_)
    names = list(model.space.structured.get_feature_names_out())
    if model.space.text is not None:
        names.extend(
            "text::" + value
            for value in model.space.text.get_feature_names_out()
        )
    contributions: list[tuple[float, str]] = []
    for index, value in zip(row.indices, row.data, strict=True):
        if index >= len(model.regressor.coef_) or index >= len(names):
            continue
        name = names[int(index)]
        if name.startswith("num::") or name.startswith("public_source::"):
            continue
        contribution = float(value) * float(model.regressor.coef_[int(index)])
        if abs(contribution) < 0.015:
            continue
        label = (
            "Description : " + name.removeprefix("text::")
            if index >= structured_size
            else _feature_label(name)
        )
        contributions.append((contribution, label))
    positives = sorted(
        [row for row in contributions if row[0] > 0],
        reverse=True,
    )[:5]
    negatives = sorted(
        [row for row in contributions if row[0] < 0],
        key=lambda row: row[0],
    )[:5]
    return (
        [
            {"label": label, "impact": round(value, 2)}
            for value, label in positives
        ],
        [
            {"label": label, "impact": round(value, 2)}
            for value, label in negatives
        ],
    )


def predict_personal_candidate(
    model: PersonalTasteModel,
    candidate: dict,
) -> dict:
    item = _candidate_item(candidate)
    if (
        model.metrics.get("active_engine") == "islands_v07"
        and model.islands is not None
    ):
        result = predict_taste_islands(model.islands, item)
        island_metrics = model.metrics["islands_v07"]
        separation = min(
            1.0,
            0.55 + abs(float(result["island_margin"])) / 100.0,
        )
        confidence = _evidence_confidence(
            float(island_metrics["mae"]),
            float(result["coverage"]) / 100.0,
            float(item.get("benchmark_reliability") or 0.0),
            separation=separation,
        )
        raw_probability = float(result["like_probability"]) / 100.0
        probability_strength = _probability_strength(
            float(result["coverage"]) / 100.0,
            float(item.get("benchmark_reliability") or 0.0),
            separation=separation,
        )
        probability = _conservative_probability(
            raw_probability,
            probability_strength,
            float(model.metrics["base_like_rate"]),
        )
        result["raw_like_probability"] = round(
            100.0 * raw_probability,
            1,
        )
        result["like_probability"] = round(100.0 * probability, 1)
        result["confidence"] = round(100.0 * confidence, 1)
        result["public_baseline_rating"] = (
            round(float(item["benchmark"]), 2)
            if item.get("benchmark") is not None
            else None
        )
        result["public_raw_rating"] = item.get("benchmark_raw")
        result["public_rating_reliability"] = round(
            100.0 * float(item.get("benchmark_reliability") or 0.0),
            1,
        )
        positive_island = result["positive_island"]
        negative_island = result["negative_island"]
        result["positive_signals"] = [
            {
                "label": "Îlot apprécié : " + positive_island["label"],
                "impact": round(result["positive_similarity"] / 100.0, 2),
            }
        ]
        result["negative_signals"] = [
            {
                "label": "Îlot moins aimé : " + negative_island["label"],
                "impact": round(-result["negative_similarity"] / 100.0, 2),
            }
        ]
        return result

    matrix, text_matrix = _transform(model.space, [item])
    coverage = _coverage(model, item, text_matrix)
    raw_residual = float(model.regressor.predict(matrix)[0])
    benchmark = _number(item.get("benchmark"))
    base = (
        float(benchmark)
        if benchmark is not None
        else model.user_baseline - model.mean_personal_residual
    )
    personal_factor = 0.45 + 0.55 * coverage
    residual = model.mean_personal_residual + (
        raw_residual - model.mean_personal_residual
    ) * personal_factor
    predicted = max(1.0, min(10.0, base + residual))
    raw_probability = float(
        model.calibrator.predict_proba(np.array([[predicted]]))[0, 1]
    )
    metrics = model.metrics["new_model"]
    mae = float(metrics["mae"])
    confidence = _evidence_confidence(
        mae,
        coverage,
        float(item.get("benchmark_reliability") or 0.0),
    )
    probability_strength = _probability_strength(
        coverage,
        float(item.get("benchmark_reliability") or 0.0),
    )
    probability = _conservative_probability(
        raw_probability,
        probability_strength,
        float(model.metrics["base_like_rate"]),
    )
    half_interval = max(
        0.5,
        model.error_p80 * (1.45 - 0.45 * confidence),
    )
    positive, negative = _learned_signals(model, item, matrix)
    return {
        "like_probability": round(100.0 * probability, 1),
        "raw_like_probability": round(100.0 * raw_probability, 1),
        "predicted_rating": round(predicted, 2),
        "prediction_low": round(max(1.0, predicted - half_interval), 1),
        "prediction_high": round(min(10.0, predicted + half_interval), 1),
        "confidence": round(100.0 * confidence, 1),
        "coverage": round(100.0 * coverage, 1),
        "public_baseline_rating": (
            round(float(benchmark), 2) if benchmark is not None else None
        ),
        "public_raw_rating": item.get("benchmark_raw"),
        "public_rating_reliability": round(
            100.0 * float(item.get("benchmark_reliability") or 0.0),
            1,
        ),
        "personal_adjustment": (
            round(predicted - float(benchmark), 2)
            if benchmark is not None
            else None
        ),
        "positive_signals": positive,
        "negative_signals": negative,
        "model_version": model.version,
        "engine": "linear_v06",
        "positive_island": None,
        "negative_island": None,
    }
