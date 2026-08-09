from __future__ import annotations

import json
import math
import sqlite3
import tempfile
import time
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import __version__
from . import personal_model as pm
from .arena import (
    ARENA_VERSION,
    _paired_comparison,
)
from .arena_protocol import (
    DEFAULT_WINDOWS,
    ChronologicalWindow,
    build_chronological_windows,
    stable_hash,
    window_manifest,
)
from .audit import _database_fingerprint, _metric_summary, _metrics
from .db import connect, initialize
from .movielens import (
    MOVIELENS_VERSION,
    MovieLensData,
    NeighborConfiguration,
    canonical_imdb_id,
    load_movielens_data,
    mapped_profile,
    neighbor_predictions,
)


MOVIELENS_ARENA_VERSION = "cineprofile-movielens-arena-2.0"
MOVIELENS_ENGINE = "movielens_user_knn_v1"
CATALOGUE_REFERENCE = "movielens_bayesian_reference"
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
NEIGHBOR_CONFIGURATIONS = (
    NeighborConfiguration(4, 10.0, 100),
    NeighborConfiguration(4, 25.0, 300),
    NeighborConfiguration(8, 10.0, 300),
    NeighborConfiguration(8, 25.0, 500),
)
ProgressCallback = Callable[[int, int, str], None]


def _latest_baseline(database: Path) -> Path:
    candidates = sorted(
        (database.parent / "logs").glob("arena_baseline_*.json"),
        reverse=True,
    )
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("arena_version") == ARENA_VERSION
            and payload.get("dense_preparation", {}).get("status") in {
                "ready",
                "not_required",
            }
            and len(payload.get("windows") or []) >= 2
        ):
            return candidate
    raise RuntimeError(
        "Aucun rapport complet de l’étape 1 n’a été trouvé dans data\\logs. "
        "Le fichier arena_baseline_20260727…json doit rester dans ce dossier."
    )


def _load_baseline(path: str | Path) -> dict:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Rapport étape 1 illisible : {target}") from exc
    if payload.get("arena_version") != ARENA_VERSION:
        raise RuntimeError(
            "Le rapport étape 1 n’utilise pas le protocole corrigé "
            f"{ARENA_VERSION}."
        )
    if not payload.get("integrity", {}).get("source_unchanged"):
        raise RuntimeError(
            "Le rapport étape 1 ne confirme pas l’intégrité de la base."
        )
    return payload


def _snapshot_database(source: Path, destination: Path) -> None:
    with closing(connect(source)) as incoming:
        with closing(sqlite3.connect(destination)) as outgoing:
            incoming.backup(outgoing)


def _reference_engine_name(baseline: dict) -> str:
    names = [
        name
        for name in (baseline.get("engine_summaries") or {})
        if name != "public_rating_reference"
    ]
    if len(names) != 1:
        raise RuntimeError(
            "Le rapport étape 1 ne contient pas exactement une référence "
            "personnelle comparable."
        )
    return names[0]


def _verify_manifests(
    items: list[dict],
    windows: list[ChronologicalWindow],
    baseline: dict,
) -> list[dict]:
    baseline_windows = baseline.get("windows") or []
    if len(windows) != len(baseline_windows):
        raise RuntimeError(
            "Le nombre de fenêtres ne correspond plus au rapport étape 1."
        )
    verified: list[dict] = []
    for window, expected in zip(windows, baseline_windows, strict=True):
        current = window_manifest(
            items,
            window,
            arena_version=ARENA_VERSION,
        )
        expected_manifest = expected.get("manifest") or {}
        if current.get("split_hash") != expected_manifest.get("split_hash"):
            raise RuntimeError(
                f"La fenêtre {window.window_id} ne correspond plus à "
                "l’étape 1. La base a probablement reçu de nouvelles notes."
            )
        verified.append(current)
    return verified


def _public_predictions(
    train: list[dict],
    test: list[dict],
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
            [pm._benchmark(item, user_mean) for item in test],
            dtype=float,
        )
        + offset,
        1.0,
        10.0,
    )


def _inner_holdout(
    items: list[dict],
    indexes: tuple[int, ...],
) -> tuple[list[dict], list[dict]]:
    ordered = [items[index] for index in indexes]
    groups: list[list[dict]] = []
    previous: str | None = None
    for item in ordered:
        current = str(item.get("date_rated") or "")
        if current != previous:
            groups.append([])
            previous = current
        groups[-1].append(item)
    target = max(20, round(len(ordered) * 0.2))
    validation: list[dict] = []
    boundary = len(groups)
    while boundary > 1 and len(validation) < target:
        boundary -= 1
        validation[0:0] = groups[boundary]
    training = [item for group in groups[:boundary] for item in group]
    if len(training) < 60 or len(validation) < 20:
        raise RuntimeError(
            "Le passé de cette fenêtre est insuffisant pour choisir les "
            "paramètres collaboratifs sans regarder la période test."
        )
    return training, validation


def _mapped_candidates(
    data: MovieLensData,
    items: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    movie_ids = np.full(len(items), -1, dtype=np.int32)
    mapped = np.zeros(len(items), dtype=bool)
    for index, item in enumerate(items):
        imdb_id = canonical_imdb_id(item.get("id"))
        movie_id = data.imdb_to_movie.get(imdb_id or "")
        if movie_id is not None and movie_id < data.centered_ratings.shape[1]:
            movie_ids[index] = movie_id
            mapped[index] = True
    return movie_ids, mapped


def _fit_calibration(
    scores: np.ndarray,
    ratings: np.ndarray,
) -> dict:
    labels = (np.asarray(ratings, dtype=float) >= pm.LIKE_THRESHOLD).astype(int)
    if len(np.unique(labels)) < 2:
        return {
            "kind": "constant",
            "probability": float(np.mean(labels)),
        }
    model = LogisticRegression(C=12.0, solver="lbfgs")
    model.fit(np.asarray(scores).reshape(-1, 1), labels)
    return {
        "kind": "logistic",
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def _calibrated_probabilities(
    calibration: dict,
    scores: np.ndarray,
) -> np.ndarray:
    if calibration["kind"] == "constant":
        return np.full(
            len(scores),
            float(calibration["probability"]),
            dtype=float,
        )
    logits = (
        float(calibration["coefficient"]) * np.asarray(scores, dtype=float)
        + float(calibration["intercept"])
    )
    logits = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _tune_collaborative_model(
    data: MovieLensData,
    items: list[dict],
    window: ChronologicalWindow,
) -> dict:
    training, validation = _inner_holdout(items, window.train_indexes)
    public = _public_predictions(training, validation)
    ratings = np.asarray(
        [float(item["rating"]) for item in validation],
        dtype=float,
    )
    movie_ids, mapped = _mapped_candidates(data, validation)
    trials: list[dict] = []
    best: tuple[tuple[float, ...], dict, np.ndarray] | None = None

    for configuration in NEIGHBOR_CONFIGURATIONS:
        collaborative = public.copy()
        diagnostics = {
            "selected_neighbors": 0,
            "positive_similarity_users": 0,
            "mapped_profile_items": 0,
        }
        if np.any(mapped):
            prediction = neighbor_predictions(
                data,
                training,
                movie_ids[mapped],
                configuration,
            )
            collaborative[mapped] = prediction.predictions
            diagnostics = {
                "selected_neighbors": prediction.selected_neighbors,
                "positive_similarity_users": (
                    prediction.positive_similarity_users
                ),
                "mapped_profile_items": prediction.mapped_profile_items,
            }
        for blend_weight in BLEND_WEIGHTS:
            blended = (
                (1.0 - blend_weight) * public
                + blend_weight * collaborative
            )
            metrics = _metrics(
                blended,
                ratings,
                predicted_ratings=blended,
            )
            row = {
                "configuration": asdict(configuration),
                "configuration_label": configuration.label,
                "collaborative_weight": blend_weight,
                "mapped_validation_items": int(np.sum(mapped)),
                **diagnostics,
                "ndcg_at_20": metrics["ndcg_at_20"],
                "precision_at_20": metrics["precision_at_20"],
                "average_rating_at_20": metrics["average_rating_at_20"],
                "mae": metrics["mae"],
            }
            trials.append(row)
            key = (
                float(metrics["ndcg_at_20"] or 0.0),
                float(metrics["precision_at_20"] or 0.0),
                float(metrics["average_rating_at_20"] or 0.0),
                -float(metrics["mae"] or 99.0),
                -blend_weight,
            )
            if best is None or key > best[0]:
                best = (key, row, blended)

    if best is None:
        raise RuntimeError("Aucun réglage collaboratif n’a pu être évalué.")
    selected = dict(best[1])
    selected["calibration"] = _fit_calibration(best[2], ratings)
    selected["inner_training_count"] = len(training)
    selected["inner_validation_count"] = len(validation)
    selected["search_space"] = {
        "neighbor_configurations": len(NEIGHBOR_CONFIGURATIONS),
        "blend_weights": list(BLEND_WEIGHTS),
        "evaluated_combinations": len(trials),
    }
    selected["trials"] = trials
    return selected


def _catalogue_candidates(
    data: MovieLensData,
    train: list[dict],
    test: list[dict],
    *,
    test_end: str,
) -> np.ndarray:
    movie_ids = data.movie_ids
    years = data.movie_years[movie_ids]
    last_year = int(str(test_end)[:4])
    eligible = (
        (data.movie_counts[movie_ids] > 0)
        & ((years == 0) | (years <= last_year))
    )
    candidates = movie_ids[eligible]
    train_ids, _ = mapped_profile(data, train)
    if len(train_ids):
        candidates = candidates[~np.isin(candidates, train_ids)]
    test_ids, mapped = _mapped_candidates(data, test)
    forced = test_ids[mapped]
    return np.unique(np.concatenate([candidates, forced])).astype(np.int32)


def _retrieval_metrics(
    candidate_ids: np.ndarray,
    scores: np.ndarray,
    relevant_ids: set[int],
) -> dict:
    relevant = relevant_ids & {int(value) for value in candidate_ids}
    result: dict[str, float | int | None] = {
        "relevant_mapped": len(relevant),
        "candidate_count": len(candidate_ids),
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
    ordered_ids = candidate_ids[order]
    ranks = {
        int(movie_id): rank
        for rank, movie_id in enumerate(ordered_ids, start=1)
        if int(movie_id) in relevant
    }
    relevant_ranks = np.asarray(list(ranks.values()), dtype=float)
    for size in (20, 50, 100, 500):
        hits = int(np.sum(relevant_ranks <= size))
        result[f"hits_at_{size}"] = hits
        result[f"recall_at_{size}"] = hits / len(relevant)
    first = float(np.min(relevant_ranks))
    result["reciprocal_first_hit"] = 1.0 / first
    result["median_relevant_rank"] = float(np.median(relevant_ranks))
    result["mean_relevant_percentile"] = float(
        np.mean(1.0 - (relevant_ranks - 1.0) / max(1, len(candidate_ids)))
    )
    return result


def _relevant_movie_ids(
    data: MovieLensData,
    test: list[dict],
    *,
    liked_only: bool,
) -> set[int]:
    result: set[int] = set()
    for item in test:
        if liked_only and float(item["rating"]) < pm.LIKE_THRESHOLD:
            continue
        imdb_id = canonical_imdb_id(item.get("id"))
        movie_id = data.imdb_to_movie.get(imdb_id or "")
        if movie_id is not None:
            result.add(movie_id)
    return result


def _catalogue_window(
    data: MovieLensData,
    train: list[dict],
    test: list[dict],
    window: ChronologicalWindow,
    selection: dict,
) -> tuple[dict, dict, dict]:
    configuration = NeighborConfiguration(**selection["configuration"])
    candidates = _catalogue_candidates(
        data,
        train,
        test,
        test_end=window.test_end,
    )
    prediction = neighbor_predictions(
        data,
        train,
        candidates,
        configuration,
    )
    weight = float(selection["collaborative_weight"])
    reference_scores = data.bayesian_scores[candidates].astype(float)
    personalized_scores = (
        (1.0 - weight) * reference_scores
        + weight * prediction.predictions
    )
    liked = _relevant_movie_ids(data, test, liked_only=True)
    watched = _relevant_movie_ids(data, test, liked_only=False)
    reference = {
        "engine": CATALOGUE_REFERENCE,
        "split": window.window_id,
        "liked_future": _retrieval_metrics(
            candidates,
            reference_scores,
            liked,
        ),
        "watched_future": _retrieval_metrics(
            candidates,
            reference_scores,
            watched,
        ),
    }
    personalized = {
        "engine": MOVIELENS_ENGINE,
        "split": window.window_id,
        "liked_future": _retrieval_metrics(
            candidates,
            personalized_scores,
            liked,
        ),
        "watched_future": _retrieval_metrics(
            candidates,
            personalized_scores,
            watched,
        ),
    }
    order = np.argsort(personalized_scores)[::-1][:20]
    examples = {
        "candidate_count": len(candidates),
        "selected_neighbors": prediction.selected_neighbors,
        "mapped_profile_items": prediction.mapped_profile_items,
        "top_20_examples": [
            {
                "movie_id": int(candidates[index]),
                "title": data.titles.get(
                    int(candidates[index]),
                    f"MovieLens {int(candidates[index])}",
                ),
                "score": round(float(personalized_scores[index]), 4),
                "future_liked_in_this_window": (
                    int(candidates[index]) in liked
                ),
                "future_watched_in_this_window": (
                    int(candidates[index]) in watched
                ),
            }
            for index in order
        ],
    }
    score_lookup = np.full(
        data.centered_ratings.shape[1],
        np.nan,
        dtype=np.float32,
    )
    support_lookup = np.zeros(
        data.centered_ratings.shape[1],
        dtype=np.int32,
    )
    score_lookup[candidates] = prediction.predictions
    support_lookup[candidates] = prediction.support_counts
    return reference, personalized, {
        **examples,
        "collaborative_score_lookup": score_lookup,
        "support_lookup": support_lookup,
    }


def _satisfaction_window(
    data: MovieLensData,
    train: list[dict],
    test: list[dict],
    window: ChronologicalWindow,
    selection: dict,
    catalogue_details: dict,
) -> tuple[dict, dict]:
    public = _public_predictions(train, test)
    movie_ids, mapped = _mapped_candidates(data, test)
    collaborative = public.copy()
    support = np.zeros(len(test), dtype=np.int32)
    lookup = catalogue_details["collaborative_score_lookup"]
    support_lookup = catalogue_details["support_lookup"]
    for index in np.flatnonzero(mapped):
        movie_id = int(movie_ids[index])
        value = float(lookup[movie_id])
        if math.isfinite(value):
            collaborative[index] = value
            support[index] = int(support_lookup[movie_id])
    weight = float(selection["collaborative_weight"])
    blended = (1.0 - weight) * public + weight * collaborative
    ratings = np.asarray([float(item["rating"]) for item in test], dtype=float)
    probabilities = _calibrated_probabilities(
        selection["calibration"],
        blended,
    )
    metrics = {
        **_metrics(
            blended,
            ratings,
            predicted_ratings=blended,
            probabilities=probabilities,
        ),
        "engine": MOVIELENS_ENGINE,
        "split": window.window_id,
    }
    order = np.argsort(blended)[::-1][:20]
    details = {
        "mapping": {
            "test_items": len(test),
            "mapped_test_items": int(np.sum(mapped)),
            "mapping_rate": round(float(np.mean(mapped)), 6),
            "mapped_with_neighbor_support": int(np.sum(support > 0)),
        },
        "selection": selection,
        "top_20_examples": [
            {
                "title": str(test[index].get("title") or ""),
                "date_rated": test[index].get("date_rated"),
                "actual_rating": float(ratings[index]),
                "predicted_rating": round(float(blended[index]), 4),
                "public_prediction": round(float(public[index]), 4),
                "collaborative_prediction": round(
                    float(collaborative[index]),
                    4,
                ),
                "neighbor_support": int(support[index]),
            }
            for index in order
        ],
    }
    return metrics, details


def _flatten_catalogue_metrics(rows: list[dict], scope: str) -> list[dict]:
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
    catalogue_rows: list[dict],
    personal_reference: str,
) -> dict:
    comparisons = {
        reference: _paired_comparison(
            satisfaction_rows,
            challenger=MOVIELENS_ENGINE,
            reference=reference,
        )
        for reference in ("public_rating_reference", personal_reference)
    }
    catalogue_flat = _flatten_catalogue_metrics(
        catalogue_rows,
        "liked_future",
    )
    catalogue_grouped: dict[str, list[dict]] = defaultdict(list)
    for row in catalogue_flat:
        catalogue_grouped[str(row["engine"])].append(row)
    catalogue_summary = {
        engine: _metric_summary(rows)
        for engine, rows in catalogue_grouped.items()
    }

    checks: list[dict] = []
    for reference, comparison in comparisons.items():
        ndcg = comparison.get("summary", {}).get("ndcg_at_20", {})
        checks.append(
            {
                "criterion": (
                    f"NDCG@20 moyen ≥ +0,02 face à {reference}"
                ),
                "passed": float(ndcg.get("mean_delta") or -99.0) >= 0.02,
                "value": ndcg.get("mean_delta"),
            }
        )
        checks.append(
            {
                "criterion": (
                    f"gain sur au moins 3/5 fenêtres face à {reference}"
                ),
                "passed": int(ndcg.get("wins") or 0) >= 3,
                "value": ndcg.get("wins"),
            }
        )
    personalized_recall = (
        catalogue_summary.get(MOVIELENS_ENGINE, {})
        .get("recall_at_100", {})
        .get("mean")
    )
    baseline_recall = (
        catalogue_summary.get(CATALOGUE_REFERENCE, {})
        .get("recall_at_100", {})
        .get("mean")
    )
    catalogue_gain = (
        float(personalized_recall) - float(baseline_recall)
        if personalized_recall is not None and baseline_recall is not None
        else None
    )
    checks.append(
        {
            "criterion": (
                "récupération des futurs 8+ au top 100 meilleure que "
                "la popularité MovieLens"
            ),
            "passed": catalogue_gain is not None and catalogue_gain > 0,
            "value": catalogue_gain,
        }
    )
    return {
        "promote_to_v1_foundation": all(check["passed"] for check in checks),
        "checks": checks,
        "satisfaction_comparisons": comparisons,
        "catalogue_liked_summaries": catalogue_summary,
    }


def _run_snapshot(
    database: Path,
    baseline: dict,
    data: MovieLensData,
    *,
    on_progress: ProgressCallback | None,
) -> dict:
    started = time.perf_counter()
    items = pm._load_training_items(database)
    requested = int(
        baseline.get("dataset", {})
        .get("date_coverage", {})
        .get("requested_windows", DEFAULT_WINDOWS)
    )
    windows, metadata = build_chronological_windows(
        items,
        requested_windows=requested,
    )
    manifests = _verify_manifests(items, windows, baseline)
    personal_reference = _reference_engine_name(baseline)
    satisfaction_rows: list[dict] = []
    catalogue_rows: list[dict] = []
    reports: list[dict] = []
    total = len(windows)

    for position, (window, manifest) in enumerate(
        zip(windows, manifests, strict=True),
        start=1,
    ):
        if on_progress:
            on_progress(
                position - 1,
                total,
                (
                    f"MovieLens · fenêtre {position}/{total} · "
                    f"{window.test_start} → {window.test_end}"
                ),
            )
        train = [items[index] for index in window.train_indexes]
        test = [items[index] for index in window.test_indexes]
        selection = _tune_collaborative_model(data, items, window)
        catalogue_reference, catalogue_personalized, catalogue_details = (
            _catalogue_window(
                data,
                train,
                test,
                window,
                selection,
            )
        )
        metrics, details = _satisfaction_window(
            data,
            train,
            test,
            window,
            selection,
            catalogue_details,
        )
        satisfaction_rows.append(metrics)
        catalogue_rows.extend(
            [catalogue_reference, catalogue_personalized]
        )
        baseline_window = baseline["windows"][position - 1]
        baseline_engines = baseline_window.get("engines") or {}
        public_metrics = baseline_engines["public_rating_reference"]["metrics"]
        personal_metrics = baseline_engines[personal_reference]["metrics"]
        satisfaction_rows.extend([public_metrics, personal_metrics])
        reports.append(
            {
                "manifest": manifest,
                "satisfaction_ranking": {
                    "public_rating_reference": {
                        "metrics": public_metrics,
                        "source": "rapport étape 1 vérifié par empreinte",
                    },
                    personal_reference: {
                        "metrics": personal_metrics,
                        "source": "rapport étape 1 vérifié par empreinte",
                    },
                    MOVIELENS_ENGINE: {
                        "metrics": metrics,
                        "details": details,
                    },
                },
                "catalogue_retrieval_positive_unlabeled": {
                    CATALOGUE_REFERENCE: catalogue_reference,
                    MOVIELENS_ENGINE: catalogue_personalized,
                    "details": {
                        key: value
                        for key, value in catalogue_details.items()
                        if not key.endswith("_lookup")
                    },
                },
            }
        )
    if on_progress:
        on_progress(total, total, "Comparaison MovieLens terminée")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in satisfaction_rows:
        grouped[str(row["engine"])].append(row)
    satisfaction_summaries = {
        engine: _metric_summary(rows) for engine, rows in grouped.items()
    }
    decision = _promotion_decision(
        satisfaction_rows,
        catalogue_rows,
        personal_reference,
    )
    mapped_all, _ = mapped_profile(data, items)
    return {
        "schema_version": 1,
        "arena_version": MOVIELENS_ARENA_VERSION,
        "baseline_arena_version": ARENA_VERSION,
        "app_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": (
            "Tester le filtrage collaboratif MovieLens comme challenger "
            "séparé, sur les mêmes cinq périodes que l’étape 1."
        ),
        "methodology": {
            "satisfaction_protocol": (
                "Les notes futures cachées et leurs empreintes sont exactement "
                "celles de l’étape 1. Les paramètres et le poids collaboratif "
                "sont choisis sur une validation interne située dans le passé."
            ),
            "catalogue_protocol": (
                "Chaque période classe les films MovieLens sortis au plus tard "
                "à sa fin et non encore notés dans le passé. Les futurs films "
                "vus ou notés 8+ sont des positifs ; les autres candidats "
                "restent non étiquetés et ne sont jamais appelés négatifs."
            ),
            "primary_metric": "ndcg_at_20",
            "catalogue_primary_metric": "recall_at_100 des futurs films notés 8+",
            "community_time_note": (
                "Comme les notes publiques de l’étape 1, les évaluations "
                "MovieLens sont celles du jeu stable disponible aujourd’hui. "
                "Elles ne sont pas tronquées à la date historique de chaque "
                "fenêtre."
            ),
        },
        "baseline_report": {
            "created_at": baseline.get("created_at"),
            "arena_version": baseline.get("arena_version"),
            "source_hash": stable_hash(baseline),
            "windows_verified": len(manifests),
            "personal_reference": personal_reference,
        },
        "dataset": {
            "movielens_version": MOVIELENS_VERSION,
            "community_users": data.user_count,
            "community_movies": data.movie_count,
            "community_ratings": data.rating_count,
            "user_history_items": len(items),
            "user_history_mapped": len(mapped_all),
            "user_history_mapping_rate": round(
                len(mapped_all) / max(1, len(items)),
                6,
            ),
            "window_metadata": metadata,
        },
        "windows": reports,
        "satisfaction_engine_summaries": satisfaction_summaries,
        "promotion_decision": decision,
        "decision": (
            "keep_for_step_3_hybrid"
            if decision["promote_to_v1_foundation"]
            else "reject_as_v1_foundation"
        ),
        "known_limitations": [
            (
                "Un film non regardé reste non étiqueté : le rappel catalogue "
                "mesure la capacité à retrouver des choix futurs connus, pas "
                "une précision absolue parmi tous les films."
            ),
            (
                "MovieLens 32M s’arrête en octobre 2023 ; les films plus récents "
                "sans correspondance utilisent la référence publique pour le "
                "test de satisfaction et sont absents du catalogue MovieLens."
            ),
            (
                "Le jeu MovieLens est autorisé pour la recherche ; cette étape "
                "est conçue pour l’usage personnel local de CineProfile."
            ),
        ],
        "privacy": (
            "Aucune note personnelle n’est envoyée à MovieLens. Le jeu public "
            "est téléchargé localement et le rapport ne contient que des "
            "métriques, des empreintes et vingt exemples par fenêtre."
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def movielens_report_path(
    database: str | Path,
    payload: dict,
) -> Path:
    target = Path(database).parent / "logs"
    target.mkdir(parents=True, exist_ok=True)
    created = datetime.fromisoformat(str(payload["created_at"]))
    stamp = created.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return target / f"arena_movielens_{stamp}.json"


def save_movielens_report(
    database: str | Path,
    payload: dict,
) -> Path:
    target = movielens_report_path(database, payload)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def run_movielens_arena(
    database: str | Path | None = None,
    *,
    baseline_report: str | Path | None = None,
    dataset_root: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict:
    source = Path(database or "data/cineprofile.db")
    initialize(source)
    baseline_path = (
        Path(baseline_report)
        if baseline_report
        else _latest_baseline(source)
    )
    baseline = _load_baseline(baseline_path)
    fingerprint_before = _database_fingerprint(source)
    root = Path(dataset_root or source.parent / "movielens")
    data = load_movielens_data(root, on_progress=on_progress)

    with tempfile.TemporaryDirectory(
        prefix="cineprofile-movielens-",
        ignore_cleanup_errors=True,
    ) as directory:
        snapshot = Path(directory) / "arena.db"
        _snapshot_database(source, snapshot)
        payload = _run_snapshot(
            snapshot,
            baseline,
            data,
            on_progress=on_progress,
        )
    fingerprint_after = _database_fingerprint(source)
    unchanged = fingerprint_before == fingerprint_after
    payload["integrity"] = {
        "source_database_snapshot_used": True,
        "source_fingerprint_before": fingerprint_before,
        "source_fingerprint_after": fingerprint_after,
        "source_unchanged": unchanged,
        "baseline_report_path": str(baseline_path),
        "movielens_files_written_outside_database": True,
    }
    if not unchanged:
        raise RuntimeError(
            "La base CineProfile a changé pendant l’étape 2. Le rapport est "
            "abandonné ; ne lance pas d’import en parallèle."
        )
    save_movielens_report(source, payload)
    return payload
