from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import Normalizer


TASTE_ISLANDS_VERSION = "cineprofile-islands-0.7.1"
POSITIVE_THRESHOLD = 8.0
NEGATIVE_THRESHOLD = 6.0


@dataclass
class IslandSpace:
    structured: DictVectorizer
    text: TfidfVectorizer | None
    reducer: TruncatedSVD | None
    positive_centroids: np.ndarray
    negative_centroids: np.ndarray
    positive_islands: list[dict]
    negative_islands: list[dict]


@dataclass
class TasteIslandsPredictor:
    version: str
    selected_alpha: float
    user_baseline: float
    mean_personal_residual: float
    space: IslandSpace
    regressor: Ridge
    calibrator: LogisticRegression
    error_p80: float
    metrics: dict


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normalise_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _category_features(item: dict) -> dict[str, float]:
    features: dict[str, float] = {}
    for dimension, entities in item.get("entities", {}).items():
        for _, name in entities:
            normalised = _normalise_name(name)
            if normalised:
                features[f"{dimension}::{normalised}"] = 1.0
    language = _normalise_name(item.get("language"))
    if language:
        features[f"language::{language}"] = 1.0
    for country in item.get("countries", []):
        normalised = _normalise_name(country)
        if normalised:
            features[f"country::{normalised}"] = 1.0
    for company in item.get("companies", [])[:8]:
        normalised = _normalise_name(company)
        if normalised:
            features[f"company::{normalised}"] = 1.0
    runtime = _number(item.get("runtime"))
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
        features[f"runtime::{band}"] = 1.0
    return features or {"unknown::unknown": 1.0}


def _text(item: dict) -> str:
    entities = item.get("entities", {})
    parts = [
        str(item.get("overview") or "").strip(),
        str(item.get("tagline") or "").strip(),
        " ".join(name for _, name in entities.get("genres", [])),
        " ".join(name for _, name in entities.get("keywords", [])),
    ]
    return " ".join(part for part in parts if part)


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


def _normalise_dense(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _fit_representation(
    items: list[dict],
) -> tuple[DictVectorizer, TfidfVectorizer | None, TruncatedSVD | None, np.ndarray]:
    structured = DictVectorizer(sparse=True, dtype=np.float64)
    structured_matrix = structured.fit_transform(
        [_category_features(item) for item in items]
    ).tocsr()
    structured_matrix = Normalizer().fit_transform(structured_matrix)

    text: TfidfVectorizer | None = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=7000,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    try:
        text_matrix = text.fit_transform([_text(item) for item in items]).tocsr()
    except ValueError:
        text = None
        text_matrix = None

    joint = (
        hstack(
            [0.38 * structured_matrix, 0.62 * text_matrix],
            format="csr",
        )
        if text_matrix is not None
        else structured_matrix
    )
    maximum_components = min(
        56,
        max(1, joint.shape[0] - 1),
        max(1, joint.shape[1] - 1),
    )
    reducer: TruncatedSVD | None
    if maximum_components >= 2:
        reducer = TruncatedSVD(
            n_components=maximum_components,
            n_iter=7,
            random_state=1979,
        )
        representation = reducer.fit_transform(joint)
    else:
        reducer = None
        representation = joint.toarray()
    return structured, text, reducer, _normalise_dense(representation)


def _transform_representation(
    space: IslandSpace,
    items: list[dict],
) -> tuple[np.ndarray, csr_matrix, csr_matrix | None]:
    structured = space.structured.transform(
        [_category_features(item) for item in items]
    ).tocsr()
    structured = Normalizer().fit_transform(structured)
    text = (
        space.text.transform([_text(item) for item in items]).tocsr()
        if space.text is not None
        else None
    )
    joint = (
        hstack([0.38 * structured, 0.62 * text], format="csr")
        if text is not None
        else structured
    )
    representation = (
        space.reducer.transform(joint)
        if space.reducer is not None
        else joint.toarray()
    )
    return _normalise_dense(representation), structured, text


def _cluster_count(size: int) -> int:
    return max(2, min(7, int(round(math.sqrt(size / 18.0)))))


def _island_terms(items: list[dict], indexes: np.ndarray) -> list[str]:
    weighted: Counter[str] = Counter()
    dimension_weights = {
        "genres": 4.0,
        "keywords": 2.5,
        "directors": 1.8,
        "writers": 1.5,
        "actors": 0.6,
        "cinematographers": 0.5,
        "composers": 0.5,
        "editors": 0.4,
    }
    for index in indexes:
        item = items[int(index)]
        for dimension, entities in item.get("entities", {}).items():
            weight = dimension_weights.get(dimension, 0.3)
            for _, name in entities:
                if name:
                    weighted[str(name)] += weight
    return [name for name, _ in weighted.most_common(4)]


def _describe_islands(
    items: list[dict],
    representation: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    polarity: str,
) -> list[dict]:
    result: list[dict] = []
    for island_id in range(len(centroids)):
        indexes = np.flatnonzero(labels == island_id)
        terms = _island_terms(items, indexes)
        similarities = representation[indexes] @ centroids[island_id]
        representatives = [
            {
                "title": items[int(indexes[position])]["title"],
                "rating": float(items[int(indexes[position])]["rating"]),
            }
            for position in np.argsort(similarities)[::-1][:3]
        ]
        result.append(
            {
                "id": island_id,
                "polarity": polarity,
                "label": " · ".join(terms[:3]) or f"Îlot {island_id + 1}",
                "terms": terms,
                "size": int(len(indexes)),
                "average_rating": round(
                    float(np.mean([items[int(index)]["rating"] for index in indexes])),
                    2,
                ),
                "representatives": representatives,
            }
        )
    return result


def _fit_centroids(
    items: list[dict],
    representation: np.ndarray,
    indexes: np.ndarray,
    polarity: str,
) -> tuple[np.ndarray, list[dict]]:
    if len(indexes) < 12:
        raise ValueError(
            f"Pas assez d’exemples {polarity} pour construire des îlots stables."
        )
    count = min(_cluster_count(len(indexes)), len(indexes))
    clustering = MiniBatchKMeans(
        n_clusters=count,
        n_init=12,
        batch_size=min(256, len(indexes)),
        random_state=1979 if polarity == "positifs" else 1980,
    )
    labels = clustering.fit_predict(representation[indexes])
    used_labels = np.unique(labels)
    centroids = _normalise_dense(
        np.vstack(
            [
                np.mean(
                    representation[indexes][labels == label],
                    axis=0,
                )
                for label in used_labels
            ]
        )
    )
    remapping = {
        int(label): new_label
        for new_label, label in enumerate(used_labels)
    }
    labels = np.array(
        [remapping[int(label)] for label in labels],
        dtype=int,
    )
    descriptions = _describe_islands(
        [items[int(index)] for index in indexes],
        representation[indexes],
        labels,
        centroids,
        polarity,
    )
    return centroids, descriptions


def fit_island_space(items: list[dict]) -> IslandSpace:
    structured, text, reducer, representation = _fit_representation(items)
    ratings = np.array([float(item["rating"]) for item in items], dtype=float)
    positive_indexes = np.flatnonzero(ratings >= POSITIVE_THRESHOLD)
    negative_indexes = np.flatnonzero(ratings <= NEGATIVE_THRESHOLD)
    positive_centroids, positive_islands = _fit_centroids(
        items,
        representation,
        positive_indexes,
        "positifs",
    )
    negative_centroids, negative_islands = _fit_centroids(
        items,
        representation,
        negative_indexes,
        "négatifs",
    )
    return IslandSpace(
        structured=structured,
        text=text,
        reducer=reducer,
        positive_centroids=positive_centroids,
        negative_centroids=negative_centroids,
        positive_islands=positive_islands,
        negative_islands=negative_islands,
    )


def transform_island_features(
    space: IslandSpace,
    items: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    representation, structured, text = _transform_representation(space, items)
    positive = representation @ space.positive_centroids.T
    negative = representation @ space.negative_centroids.T
    top_positive = np.max(positive, axis=1)
    top_negative = np.max(negative, axis=1)
    second_positive = (
        np.partition(positive, -2, axis=1)[:, -2]
        if positive.shape[1] > 1
        else top_positive
    )
    second_negative = (
        np.partition(negative, -2, axis=1)[:, -2]
        if negative.shape[1] > 1
        else top_negative
    )
    metadata = np.array(
        [
            [
                ((_number(item.get("benchmark")) or 7.0) - 7.0) / 2.0,
                math.log1p(max(0.0, _number(item.get("votes")) or 0.0)) / 14.0,
                ((_number(item.get("runtime")) or 110.0) - 110.0) / 80.0,
            ]
            for item in items
        ],
        dtype=float,
    )
    features = np.column_stack(
        [
            positive,
            negative,
            top_positive,
            second_positive,
            top_negative,
            second_negative,
            top_positive - top_negative,
            metadata,
        ]
    )
    structured_names = set(space.structured.vocabulary_)
    evidence: list[dict] = []
    for index, item in enumerate(items):
        categories = list(_category_features(item))
        known_categories = sum(name in structured_names for name in categories)
        structured_coverage = (
            known_categories / len(categories) if categories else 0.15
        )
        text_coverage = (
            min(1.0, float(text.getrow(index).nnz) / 14.0)
            if text is not None
            else 0.0
        )
        description_coverage = _description_coverage(item)
        positive_id = int(np.argmax(positive[index]))
        negative_id = int(np.argmax(negative[index]))
        evidence.append(
            {
                "coverage": max(
                    0.10,
                    min(
                        1.0,
                        0.50 * structured_coverage
                        + 0.25 * text_coverage
                        + 0.25 * description_coverage,
                    ),
                ),
                "positive_similarity": float(top_positive[index]),
                "negative_similarity": float(top_negative[index]),
                "margin": float(top_positive[index] - top_negative[index]),
                "positive_island": space.positive_islands[positive_id],
                "negative_island": space.negative_islands[negative_id],
            }
        )
    return features, evidence


def predict_taste_islands(
    model: TasteIslandsPredictor,
    item: dict,
) -> dict:
    features, evidence_rows = transform_island_features(model.space, [item])
    evidence = evidence_rows[0]
    raw_residual = float(model.regressor.predict(features)[0])
    benchmark = _number(item.get("benchmark"))
    base = (
        float(benchmark)
        if benchmark is not None
        else model.user_baseline - model.mean_personal_residual
    )
    coverage = float(evidence["coverage"])
    personal_factor = 0.45 + 0.55 * coverage
    residual = model.mean_personal_residual + (
        raw_residual - model.mean_personal_residual
    ) * personal_factor
    predicted = max(1.0, min(10.0, base + residual))
    probability = float(
        model.calibrator.predict_proba(np.array([[predicted]]))[0, 1]
    )
    mae = float(model.metrics["mae"])
    reliability = max(0.35, min(0.92, 1.0 - mae / 3.2))
    votes = max(0.0, float(item.get("votes") or 0.0))
    public_coverage = 0.55 + 0.45 * min(1.0, math.log1p(votes) / 8.0)
    separation = min(1.0, 0.55 + abs(float(evidence["margin"])))
    confidence = (
        reliability
        * (0.45 + 0.55 * coverage)
        * public_coverage
        * separation
    )
    half_interval = max(
        0.5,
        model.error_p80 * (1.35 - 0.35 * coverage),
    )
    return {
        "like_probability": round(100.0 * probability, 1),
        "predicted_rating": round(predicted, 2),
        "prediction_low": round(max(1.0, predicted - half_interval), 1),
        "prediction_high": round(min(10.0, predicted + half_interval), 1),
        "confidence": round(100.0 * confidence, 1),
        "coverage": round(100.0 * coverage, 1),
        "public_baseline_rating": (
            round(float(benchmark), 2) if benchmark is not None else None
        ),
        "personal_adjustment": (
            round(predicted - float(benchmark), 2)
            if benchmark is not None
            else None
        ),
        "positive_island": evidence["positive_island"],
        "negative_island": evidence["negative_island"],
        "positive_similarity": round(
            100.0 * float(evidence["positive_similarity"]),
            1,
        ),
        "negative_similarity": round(
            100.0 * float(evidence["negative_similarity"]),
            1,
        ),
        "island_margin": round(100.0 * float(evidence["margin"]), 1),
        "model_version": model.version,
        "engine": "islands_v07",
    }
