from __future__ import annotations

import gc
import json
import logging
import sqlite3
import tempfile
import time
from collections import defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import __version__
from . import hybrid_model as hm
from . import personal_model as pm
from .audit import (
    _database_fingerprint,
    _evaluate_split,
    _metric_summary,
    _metrics,
)
from .arena_protocol import (
    DEFAULT_WINDOWS,
    ChronologicalWindow,
    build_chronological_windows,
    rated_date,
    stable_hash,
    window_manifest,
)
from .db import connect, initialize
from .diagnostics import configure_logging


ARENA_SCHEMA_VERSION = 1
ARENA_VERSION = "cineprofile-offline-arena-1.1"
ProgressCallback = Callable[[int, int, str], None]


def _public_reference(
    train: list[dict],
    test: list[dict],
    *,
    split_name: str,
) -> tuple[dict, dict]:
    train_ratings = np.asarray(
        [float(item["rating"]) for item in train],
        dtype=float,
    )
    test_ratings = np.asarray(
        [float(item["rating"]) for item in test],
        dtype=float,
    )
    user_mean = float(np.mean(train_ratings))
    train_public = np.asarray(
        [pm._benchmark(item, user_mean) for item in train],
        dtype=float,
    )
    residual = float(np.mean(train_ratings - train_public))
    train_predictions = np.clip(train_public + residual, 1.0, 10.0)
    test_predictions = np.clip(
        np.asarray(
            [pm._benchmark(item, user_mean) for item in test],
            dtype=float,
        )
        + residual,
        1.0,
        10.0,
    )
    train_labels = (train_ratings >= pm.LIKE_THRESHOLD).astype(int)
    if len(np.unique(train_labels)) >= 2:
        calibrator = LogisticRegression(C=12.0, solver="lbfgs")
        calibrator.fit(train_predictions.reshape(-1, 1), train_labels)
        probabilities = calibrator.predict_proba(test_predictions.reshape(-1, 1))[:, 1]
    else:
        probabilities = np.full(
            len(test),
            float(np.mean(train_labels)),
            dtype=float,
        )
    metrics = {
        **_metrics(
            test_predictions,
            test_ratings,
            predicted_ratings=test_predictions,
            probabilities=probabilities,
        ),
        "engine": "public_rating_reference",
        "split": split_name,
    }
    order = np.argsort(test_predictions)[::-1][:20]
    details = {
        "training_user_mean": round(user_mean, 4),
        "training_personal_offset": round(residual, 4),
        "top_20_examples": [
            {
                "title": str(test[int(index)].get("title") or ""),
                "date_rated": test[int(index)].get("date_rated"),
                "actual_rating": float(test_ratings[int(index)]),
                "predicted_rating": round(
                    float(test_predictions[int(index)]),
                    4,
                ),
                "score": round(float(test_predictions[int(index)]), 4),
            }
            for index in order
        ],
    }
    return metrics, details


def _active_reference(
    items: list[dict],
    dense_embeddings: np.ndarray | None,
    window: ChronologicalWindow,
    configuration: dict,
) -> tuple[dict, dict]:
    train_indexes = np.asarray(window.train_indexes, dtype=int)
    test_indexes = np.asarray(window.test_indexes, dtype=int)
    train = [items[int(index)] for index in train_indexes]
    test = [items[int(index)] for index in test_indexes]
    engine = str(configuration.get("engine") or "")
    if engine in {"personal_v09", "hybrid_v08"}:
        settings = configuration.get("configuration") or {}
        variant = str(
            configuration.get("variant")
            or settings.get("variant")
            or hm.DEFAULT_PERSONAL_VARIANT
        )
        blocks = tuple(hm.HYBRID_VARIANTS[variant]["blocks"])
        train_dense = dense_embeddings[train_indexes] if "dense" in blocks else None
        test_dense = dense_embeddings[test_indexes] if "dense" in blocks else None
        return hm.evaluate_hybrid_split(
            train,
            test,
            train_dense,
            test_dense,
            variant=variant,
            split_name=window.window_id,
        )

    rows, details = _evaluate_split(
        items,
        train_indexes,
        test_indexes,
        split_name=window.window_id,
    )
    wanted = "islands_v07" if engine == "islands_v07" else "linear_v06"
    selected = next(
        (row for row in rows if row.get("engine") == wanted),
        None,
    )
    if selected is None:
        raise RuntimeError(
            f"Le moteur de référence {wanted} n’a produit aucune mesure."
        )
    top_examples = details.get("top_20_examples", {}).get(wanted, [])
    return selected, {"top_20_examples": top_examples}


def _active_reference_description(
    database: str | Path | None,
) -> dict:
    configuration = hm.active_configuration(database) or {}
    engine = str(configuration.get("engine") or "linear_v06")
    settings = configuration.get("configuration") or {}
    variant = (
        str(settings.get("variant") or hm.DEFAULT_PERSONAL_VARIANT)
        if engine in {"personal_v09", "hybrid_v08"}
        else None
    )
    return {
        "engine": engine,
        "variant": variant,
        "variant_label": (
            str(hm.HYBRID_VARIANTS[variant]["label"])
            if variant in hm.HYBRID_VARIANTS
            else None
        ),
        "dense_required": bool(
            variant and "dense" in hm.HYBRID_VARIANTS[variant]["blocks"]
        ),
        "source": "configuration active de CineProfile 0.9.1",
        "selection_leakage_note": (
            "Cette référence peut avoir été choisie par un ancien audit sur "
            "l’historique complet. Elle sert de point de comparaison, pas de "
            "preuve finale pour une future version."
        ),
    }


def _paired_comparison(
    rows: list[dict],
    *,
    challenger: str,
    reference: str,
) -> dict:
    by_split: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_split[str(row["split"])][str(row["engine"])] = row
    metrics = (
        "ndcg_at_20",
        "precision_at_20",
        "average_rating_at_20",
        "mae",
        "brier",
    )
    paired: list[dict] = []
    for split_name in sorted(by_split):
        current = by_split[split_name]
        if challenger not in current or reference not in current:
            continue
        deltas: dict[str, float] = {}
        for metric in metrics:
            left = current[challenger].get(metric)
            right = current[reference].get(metric)
            if left is None or right is None:
                continue
            deltas[metric] = round(float(left) - float(right), 6)
        paired.append({"split": split_name, "deltas": deltas})
    summary: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [row["deltas"][metric] for row in paired if metric in row["deltas"]]
        if values:
            summary[metric] = {
                "mean_delta": round(float(np.mean(values)), 6),
                "minimum_delta": round(float(np.min(values)), 6),
                "maximum_delta": round(float(np.max(values)), 6),
                "wins": int(
                    sum(
                        value > 0 if metric not in {"mae", "brier"} else value < 0
                        for value in values
                    )
                ),
                "comparisons": len(values),
            }
    return {
        "challenger": challenger,
        "reference": reference,
        "paired_windows": paired,
        "summary": summary,
    }


def _dataset_summary(
    items: list[dict],
    window_metadata: dict,
) -> dict:
    ratings = np.asarray([float(item["rating"]) for item in items], dtype=float)
    dated = [
        rated_date(item).isoformat() for item in items if rated_date(item) is not None
    ]
    enriched = sum(bool(str(item.get("overview") or "").strip()) for item in items)
    return {
        "usable_rated_films": len(items),
        "positive_definition": f"note IMDb ≥ {pm.LIKE_THRESHOLD:g}/10",
        "positive_count": int(np.sum(ratings >= pm.LIKE_THRESHOLD)),
        "positive_rate": round(
            float(np.mean(ratings >= pm.LIKE_THRESHOLD)),
            6,
        ),
        "mean_rating": round(float(np.mean(ratings)), 6),
        "date_coverage": {
            "first": min(dated),
            "last": max(dated),
            **window_metadata,
        },
        "metadata_coverage": {
            "overview_count": enriched,
            "overview_rate": round(enriched / max(1, len(items)), 6),
        },
    }


def _run_snapshot(
    database: str | Path | None,
    *,
    requested_windows: int,
    model_cache_directory: str | Path | None,
    on_progress: ProgressCallback | None,
    logger: logging.Logger,
) -> dict:
    started = time.perf_counter()
    items = pm._load_training_items(database)
    windows, window_metadata = build_chronological_windows(
        items,
        requested_windows=requested_windows,
    )
    active = _active_reference_description(database)
    dense_embeddings: np.ndarray | None = None
    dense_preparation = {
        "required": bool(active["dense_required"]),
        "status": "not_required",
        "source_database_written": False,
        "snapshot_database_written": False,
    }
    phase_offset = 0
    if active["dense_required"]:
        phase_offset = 1
        try:
            coverage_before = hm.dense_embedding_cache_coverage(
                database,
                items,
            )

            def semantic_progress(
                current: int,
                total: int,
                message: str,
            ) -> None:
                if not on_progress:
                    return
                percent = (
                    100
                    if total <= 0
                    else min(100, round(100 * current / total))
                )
                on_progress(
                    0,
                    len(windows) + phase_offset,
                    f"Préparation sémantique · {percent}% · {message}",
                )

            dense_embeddings = hm.prepare_dense_embeddings(
                database,
                items,
                cache_directory=model_cache_directory,
                on_progress=semantic_progress,
            )
            coverage_after = hm.dense_embedding_cache_coverage(
                database,
                items,
            )
            if coverage_after["missing"]:
                raise RuntimeError(
                    "Le cache temporaire reste incomplet "
                    f"({coverage_after['missing']} vecteurs manquants)."
                )
            dense_preparation = {
                "required": True,
                "status": "ready",
                "total_vectors": coverage_after["total"],
                "cached_before": coverage_before["cached"],
                "generated_on_snapshot": coverage_before["missing"],
                "missing_after": coverage_after["missing"],
                "model_cache_reused": True,
                "source_database_written": False,
                "snapshot_database_written": bool(
                    coverage_before["missing"]
                ),
            }
            if on_progress:
                on_progress(
                    1,
                    len(windows) + phase_offset,
                    (
                        "Sémantique prête · "
                        f"{coverage_after['total']} vecteurs disponibles"
                    ),
                )
        except Exception as exc:
            logger.exception(
                "arena_dense_reference_preparation_failed | variant=%s",
                active.get("variant"),
            )
            raise RuntimeError(
                "Impossible de préparer la référence sémantique personnelle. "
                "Aucune mesure de remplacement n’a été produite : "
                f"{exc}"
            ) from exc

    total_steps = len(windows) + phase_offset
    all_rows: list[dict] = []
    split_reports: list[dict] = []
    active_engine_name: str | None = None
    active_completed = 0
    for position, window in enumerate(windows, start=1):
        message = (
            f"Fenêtre {position}/{len(windows)} · "
            f"{window.test_start} → {window.test_end}"
        )
        if on_progress:
            on_progress(
                phase_offset + position - 1,
                total_steps,
                message,
            )
        train = [items[index] for index in window.train_indexes]
        test = [items[index] for index in window.test_indexes]
        public_metrics, public_details = _public_reference(
            train,
            test,
            split_name=window.window_id,
        )
        all_rows.append(public_metrics)
        engines = {
            "public_rating_reference": {
                "metrics": public_metrics,
                "details": public_details,
            }
        }
        try:
            active_metrics, active_details = _active_reference(
                items,
                dense_embeddings,
                window,
                active,
            )
            active_engine_name = str(active_metrics["engine"])
            active_completed += 1
            all_rows.append(active_metrics)
            engines[active_engine_name] = {
                "metrics": active_metrics,
                "details": active_details,
            }
        except Exception as exc:
            logger.exception(
                "arena_active_reference_failed | split=%s",
                window.window_id,
            )
            raise RuntimeError(
                "La référence personnelle a échoué sur "
                f"{window.window_id}. Le rapport incomplet est abandonné : "
                f"{exc}"
            ) from exc
        split_report = {
            "manifest": window_manifest(
                items,
                window,
                arena_version=ARENA_VERSION,
            ),
            "engines": engines,
        }
        split_reports.append(split_report)
        logger.info(
            "arena_window_completed | split=%s | train=%s | test=%s | "
            "public_ndcg20=%s | active_engine=%s | active_ndcg20=%s",
            window.window_id,
            len(train),
            len(test),
            public_metrics.get("ndcg_at_20"),
            active_engine_name,
            (
                engines.get(active_engine_name, {}).get("metrics", {}).get("ndcg_at_20")
                if active_engine_name
                else None
            ),
        )

    if not active_engine_name or active_completed != len(windows):
        raise RuntimeError(
            "La référence personnelle n’a pas été mesurée sur toutes les "
            "fenêtres. Le rapport incomplet est abandonné."
        )
    if on_progress:
        on_progress(total_steps, total_steps, "Rapport terminé")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        grouped[str(row["engine"])].append(row)
    summaries = {engine: _metric_summary(rows) for engine, rows in grouped.items()}
    paired = (
        _paired_comparison(
            all_rows,
            challenger=active_engine_name,
            reference="public_rating_reference",
        )
        if active_engine_name
        else None
    )
    return {
        "schema_version": ARENA_SCHEMA_VERSION,
        "arena_version": ARENA_VERSION,
        "app_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": (
            "Geler des cas chronologiques reproductibles et mesurer les "
            "moteurs futurs sur exactement les mêmes notes cachées, sans "
            "modifier CineProfile ni apprendre à partir du futur."
        ),
        "methodology": {
            "protocol": "fenêtres chronologiques expansives et disjointes",
            "training_rule": (
                "À chaque fenêtre, le modèle ne voit que les notes antérieures "
                "à la période test. Ses hyperparamètres sont choisis uniquement "
                "dans ce passé."
            ),
            "same_day_rule": (
                "Toutes les notes d’une même date restent dans le même groupe "
                "pour empêcher une séparation arbitraire d’une session."
            ),
            "test_reuse": ("Chaque note testée n’apparaît que dans une seule fenêtre."),
            "primary_metric": "ndcg_at_20",
            "secondary_metrics": [
                "precision_at_20",
                "average_rating_at_20",
                "mae",
                "brier",
            ],
            "unknown_films": (
                "Les films jamais notés ne sont ni positifs ni négatifs dans "
                "cette étape."
            ),
        },
        "dataset": _dataset_summary(items, window_metadata),
        "reference_engine": active,
        "dense_preparation": dense_preparation,
        "benchmark_status": {
            "satisfaction_ranking": {
                "status": "ready",
                "scope": (
                    "Classement de films réellement notés mais cachés dans "
                    "leur période chronologique."
                ),
            },
            "catalogue_retrieval": {
                "status": "pending_step_2",
                "scope": (
                    "Retrouver de bons candidats parmi un catalogue externe "
                    "contenant des films non vus."
                ),
                "reason": (
                    "MovieLens et son mapping IMDb/TMDB ne sont pas encore "
                    "branchés. Inventer des négatifs parmi les films non notés "
                    "biaiserait le résultat."
                ),
            },
        },
        "windows": split_reports,
        "engine_summaries": summaries,
        "reference_vs_public": paired,
        "reference_failures": [],
        "dense_reference_error": None,
        "promotion_gate_for_future_models": {
            "note": (
                "Ces seuils sont des critères d’acceptation, pas des poids du moteur."
            ),
            "requirements": [
                "delta moyen de NDCG@20 ≥ +0,02 face à la référence active",
                "gain sur au moins 60 % des fenêtres comparables",
                "régression NDCG@20 sur la fenêtre la plus récente ≥ -0,02",
                "note moyenne du top 20 non dégradée",
                "MAE non dégradée de plus de 0,10 point",
                ("preuve séparée sur la récupération catalogue à partir de l’étape 2"),
            ],
        },
        "future_challengers": {
            "movielens_collaborative": "pending_step_2",
            "multilingual_e5": "pending_step_3",
            "bge_m3": "pending_step_3",
        },
        "known_limitations": [
            (
                "Cette étape mesure la satisfaction parmi des films déjà "
                "notés ; elle ne mesure pas encore la découverte catalogue."
            ),
            (
                "Les métadonnées et notes publiques sont celles disponibles "
                "aujourd’hui, pas nécessairement celles du jour de notation."
            ),
            (
                "La configuration 0.9.1 active est une référence historique ; "
                "ses règles d’envie codées en dur ne font pas partie de cette "
                "arène et ne seront pas conservées dans le moteur v1."
            ),
        ],
        "privacy": (
            "Le rapport contient des métriques, des empreintes de découpages "
            "et au maximum vingt exemples par moteur et par fenêtre. Il "
            "n’exporte ni jeton TMDB ni historique intégral."
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def arena_report_path(
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
    return target / f"arena_baseline_{stamp}.json"


def save_arena_report(
    database: str | Path | None,
    payload: dict,
) -> Path:
    path = arena_report_path(database, payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    logging.getLogger("cineprofile").info(
        "arena_report_saved | file=%s",
        path.name,
    )
    return path


def _snapshot_database(
    database: str | Path | None,
    destination: Path,
) -> None:
    with closing(connect(database)) as source:
        with closing(sqlite3.connect(destination)) as target:
            source.backup(target)


def run_offline_arena(
    database: str | Path | None = None,
    *,
    requested_windows: int = DEFAULT_WINDOWS,
    on_progress: ProgressCallback | None = None,
) -> dict:
    original = Path(database or "data/cineprofile.db")
    initialize(original)
    logger = configure_logging(original)
    fingerprint_before = _database_fingerprint(original)
    logger.info(
        "arena_started | version=%s | requested_windows=%s",
        ARENA_VERSION,
        requested_windows,
    )
    with tempfile.TemporaryDirectory(
        prefix="cineprofile-arena-",
        ignore_cleanup_errors=True,
    ) as directory:
        snapshot = Path(directory) / "arena.db"
        _snapshot_database(original, snapshot)
        payload = _run_snapshot(
            snapshot,
            requested_windows=requested_windows,
            model_cache_directory=original.parent / "models",
            on_progress=on_progress,
            logger=logger,
        )
        fingerprint_after = _database_fingerprint(original)
        unchanged = fingerprint_before == fingerprint_after
        payload["integrity"] = {
            "source_database_snapshot_used": True,
            "source_fingerprint_before": fingerprint_before,
            "source_fingerprint_after": fingerprint_after,
            "source_unchanged": unchanged,
            "semantic_vectors_written_to_source_database": False,
        }
        if not unchanged:
            logger.error("arena_integrity_failed | source database changed during run")
            raise RuntimeError(
                "La base a changé pendant le laboratoire. Le résultat est "
                "abandonné ; relance-le sans importer ni enrichir en parallèle."
            )
        save_arena_report(original, payload)
        gc.collect()
    logger.info(
        "arena_completed | windows=%s | elapsed_seconds=%s",
        len(payload["windows"]),
        payload["elapsed_seconds"],
    )
    return payload
