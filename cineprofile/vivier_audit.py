from __future__ import annotations

import gc
import json
import sqlite3
import tempfile
import time
from collections import defaultdict
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Callable

from . import __version__
from .arena_protocol import (
    DEFAULT_WINDOWS,
    ChronologicalWindow,
    build_chronological_windows,
    stable_hash,
    window_manifest,
)
from .audit import _database_fingerprint
from .candidate_pool import (
    SOURCE_PRIORITY,
    SOURCE_SEMANTIC,
    balanced_candidate_order,
    build_candidate_pool,
)
from .db import connect, initialize
from .diagnostics import configure_logging
from .personal_model import LIKE_THRESHOLD, _load_training_items
from .profile import build_profile
from .recommender import SEARCH_DEPTHS


VIVIER_AUDIT_SCHEMA_VERSION = 1
VIVIER_AUDIT_VERSION = "cineprofile-vivier-audit-1.0"
DEFAULT_BUDGETS = (100, 300, 500)
DEFAULT_PERIOD_YEARS = 3
ProgressCallback = Callable[[int, int, str], None]


def _snapshot_database(source: Path, destination: Path) -> None:
    with closing(connect(source)) as incoming:
        with closing(sqlite3.connect(destination)) as outgoing:
            incoming.backup(outgoing)


def _training_snapshot(
    source: Path,
    destination: Path,
    train_imdb_ids: set[str],
) -> None:
    _snapshot_database(source, destination)
    with connect(destination) as connection:
        connection.execute(
            "CREATE TEMP TABLE audit_train_ids(imdb_id TEXT PRIMARY KEY)"
        )
        connection.executemany(
            "INSERT INTO audit_train_ids(imdb_id) VALUES (?)",
            [(imdb_id,) for imdb_id in sorted(train_imdb_ids)],
        )
        connection.execute(
            """
            DELETE FROM titles
            WHERE imdb_id NOT IN (SELECT imdb_id FROM audit_train_ids)
            """
        )
        # These rows describe decisions or learned state made after the
        # historical cutoff. They must not influence the reconstructed past.
        for table in (
            "recommendation_feedback",
            "profile_preferences",
            "profile_runs",
            "personal_models",
            "active_model_configuration",
        ):
            connection.execute(f"DELETE FROM {table}")


def _rated_items(database: str | Path | None) -> list[dict]:
    items = _load_training_items(database)
    with connect(database) as connection:
        mappings = {
            str(row["imdb_id"]): {
                "tmdb_id": (
                    int(row["tmdb_id"])
                    if row["tmdb_id"] is not None
                    else None
                ),
                "release_date": row["release_date"],
            }
            for row in connection.execute(
                "SELECT imdb_id, tmdb_id, release_date FROM titles"
            )
        }
    for item in items:
        mapping = mappings.get(str(item["id"]), {})
        item["tmdb_id"] = mapping.get("tmdb_id")
        item["release_date"] = (
            item.get("release_date") or mapping.get("release_date")
        )
    return items


def _period_start(end_date: str, years: int) -> str:
    end = date.fromisoformat(end_date[:10])
    try:
        return end.replace(year=end.year - years).isoformat()
    except ValueError:
        return end.replace(year=end.year - years, day=28).isoformat()


def _retrieval_metrics(
    candidates: list[dict],
    targets: list[dict],
    *,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
) -> dict[str, float | int | None]:
    positions = {
        int(candidate["id"]): position
        for position, candidate in enumerate(candidates, start=1)
    }
    target_ids = {
        int(target["tmdb_id"])
        for target in targets
        if target.get("tmdb_id") is not None
    }
    result: dict[str, float | int | None] = {
        "target_count": len(target_ids),
        "candidate_count": len(candidates),
        "present_anywhere": sum(value in positions for value in target_ids),
    }
    for budget in budgets:
        hits = sum(
            positions.get(target_id, budget + 1) <= budget
            for target_id in target_ids
        )
        result[f"hits_at_{budget}"] = hits
        result[f"recall_at_{budget}"] = (
            hits / len(target_ids) if target_ids else None
        )
    return result


def _without_source(candidates: list[dict], removed_source: str) -> list[dict]:
    remaining: list[dict] = []
    for candidate in candidates:
        acquisition_sources = [
            source
            for source in candidate.get("_sources", [])
            if source != SOURCE_SEMANTIC and source != removed_source
        ]
        if not acquisition_sources:
            continue
        sources = list(acquisition_sources)
        if SOURCE_SEMANTIC in candidate.get("_sources", []):
            sources.append(SOURCE_SEMANTIC)
        remaining.append({**candidate, "_sources": sources})
    return balanced_candidate_order(remaining)


def evaluate_vivier_pool(
    candidates: list[dict],
    targets: list[dict],
    *,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
) -> dict:
    """Measure pool recall and cached source ablations without ranking films."""
    baseline = _retrieval_metrics(candidates, targets, budgets=budgets)
    target_ids = {
        int(target["tmdb_id"])
        for target in targets
        if target.get("tmdb_id") is not None
    }
    source_coverage: dict[str, int] = {}
    for source in SOURCE_PRIORITY:
        if source == SOURCE_SEMANTIC:
            continue
        source_ids = {
            int(candidate["id"])
            for candidate in candidates
            if source in candidate.get("_sources", [])
        }
        source_coverage[source] = len(target_ids & source_ids)
    ablations = {
        source: _retrieval_metrics(
            _without_source(candidates, source),
            targets,
            budgets=budgets,
        )
        for source in source_coverage
    }
    return {
        "baseline": baseline,
        "source_target_coverage": source_coverage,
        "ablations": ablations,
    }


def _target_rows(
    items: list[dict],
    window: ChronologicalWindow,
) -> tuple[list[dict], int]:
    liked = [
        items[index]
        for index in window.test_indexes
        if float(items[index]["rating"]) >= LIKE_THRESHOLD
    ]
    measurable = [item for item in liked if item.get("tmdb_id") is not None]
    return measurable, len(liked) - len(measurable)


def _within_period(item: dict, start_date: str, end_date: str) -> bool:
    raw = str(item.get("release_date") or "")[:10]
    try:
        released = date.fromisoformat(raw)
    except ValueError:
        return False
    return date.fromisoformat(start_date) <= released <= date.fromisoformat(
        end_date
    )


def _exclude_training_history(
    candidates: list[dict],
    database: str | Path,
    diagnostics: dict[str, object],
) -> list[dict]:
    """Mirror the production exclusion of films already present in history."""
    with connect(database) as connection:
        seen_tmdb = {
            int(row[0])
            for row in connection.execute(
                "SELECT tmdb_id FROM titles WHERE tmdb_id IS NOT NULL"
            )
        }
    unseen = [
        candidate
        for candidate in candidates
        if int(candidate["id"]) not in seen_tmdb
    ]
    diagnostics["excluded_training_history"] = len(candidates) - len(unseen)
    diagnostics["after_training_history_exclusion"] = len(unseen)
    trace = diagnostics.get("candidate_trace")
    if isinstance(trace, dict):
        ranks = {
            int(candidate["id"]): position
            for position, candidate in enumerate(unseen, start=1)
        }
        for candidate_id, item in trace.items():
            if isinstance(item, dict) and item.get("state") == "eligible":
                item["rank"] = ranks.get(int(candidate_id))
    return unseen


def _missing_traces(
    targets: list[dict],
    diagnostics: dict[str, object],
    *,
    budgets: tuple[int, ...],
) -> list[dict]:
    trace = diagnostics.get("candidate_trace") or {}
    rows: list[dict] = []
    largest = max(budgets)
    for target in targets:
        tmdb_id = int(target["tmdb_id"])
        item = dict(trace.get(str(tmdb_id)) or {})
        rank = item.get("rank")
        if rank is not None and int(rank) <= largest:
            continue
        state = str(item.get("state") or "absent_from_all_sources")
        if rank is not None:
            state = "beyond_analysis_budget"
        rows.append(
            {
                "tmdb_id": tmdb_id,
                "imdb_id": target.get("id"),
                "title": target.get("title"),
                "rating": float(target["rating"]),
                "date_rated": target.get("date_rated"),
                "release_date": target.get("release_date"),
                "state": state,
                "rank": rank,
                "sources": item.get("sources", []),
                "vote_count": item.get("vote_count"),
            }
        )
    return rows


def _aggregate_windows(
    windows: list[dict],
    *,
    budgets: tuple[int, ...],
) -> dict:
    target_count = sum(
        int(window["metrics"]["baseline"]["target_count"])
        for window in windows
    )
    eligible_target_count = sum(
        int(window["eligible_period_metrics"]["target_count"])
        for window in windows
    )
    summary: dict[str, object] = {
        "measurable_liked_films": target_count,
        "liked_without_tmdb_id": sum(
            int(window["liked_without_tmdb_id"]) for window in windows
        ),
        "eligible_in_default_period": eligible_target_count,
        "missing_or_beyond_500": sum(
            len(window["missing_traces"]) for window in windows
        ),
    }
    for budget in budgets:
        hits = sum(
            int(window["metrics"]["baseline"][f"hits_at_{budget}"])
            for window in windows
        )
        eligible_hits = sum(
            int(window["eligible_period_metrics"][f"hits_at_{budget}"])
            for window in windows
        )
        summary[f"hits_at_{budget}"] = hits
        summary[f"recall_at_{budget}"] = (
            hits / target_count if target_count else None
        )
        summary[f"eligible_recall_at_{budget}"] = (
            eligible_hits / eligible_target_count
            if eligible_target_count
            else None
        )

    coverage: defaultdict[str, int] = defaultdict(int)
    source_counts: defaultdict[str, int] = defaultdict(int)
    for window in windows:
        for source, count in window["metrics"][
            "source_target_coverage"
        ].items():
            coverage[source] += int(count)
        for source, count in window["source_counts"].items():
            source_counts[source] += int(count)
    summary["source_target_coverage"] = dict(coverage)
    summary["source_candidate_counts"] = dict(source_counts)

    ablation_rows: list[dict] = []
    for source in sorted(coverage):
        row: dict[str, object] = {
            "source": source,
            "targets_found_by_source": coverage[source],
            "raw_candidates": source_counts.get(source, 0),
        }
        for budget in budgets:
            without_hits = sum(
                int(
                    window["metrics"]["ablations"][source][
                        f"hits_at_{budget}"
                    ]
                )
                for window in windows
            )
            baseline_hits = int(summary[f"hits_at_{budget}"])
            row[f"hits_without_source_at_{budget}"] = without_hits
            row[f"lost_hits_at_{budget}"] = baseline_hits - without_hits
            row[f"recall_without_source_at_{budget}"] = (
                without_hits / target_count if target_count else None
            )
        ablation_rows.append(row)
    summary["source_ablation"] = sorted(
        ablation_rows,
        key=lambda row: (
            int(row.get("lost_hits_at_500") or 0),
            int(row.get("targets_found_by_source") or 0),
        ),
        reverse=True,
    )
    return summary


def run_vivier_audit(
    client,
    database: str | Path | None = None,
    *,
    requested_windows: int = DEFAULT_WINDOWS,
    period_years: int = DEFAULT_PERIOD_YEARS,
    depth: str = "Normale",
    reliability: str = "Forte",
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    on_progress: ProgressCallback | None = None,
) -> dict:
    original = Path(database or "data/cineprofile.db")
    initialize(original)
    if depth not in SEARCH_DEPTHS:
        raise ValueError(f"Profondeur inconnue : {depth}")
    budgets = tuple(sorted({max(1, int(value)) for value in budgets}))
    period_years = max(1, int(period_years))
    logger = configure_logging(original)
    fingerprint_before = _database_fingerprint(original)
    started = time.perf_counter()
    items = _rated_items(original)
    chronological, window_metadata = build_chronological_windows(
        items,
        requested_windows=requested_windows,
    )
    reports: list[dict] = []

    with tempfile.TemporaryDirectory(
        prefix="cineprofile-vivier-audit-",
        ignore_cleanup_errors=True,
    ) as directory:
        root = Path(directory)
        for position, window in enumerate(chronological, start=1):
            if on_progress:
                on_progress(
                    position - 1,
                    len(chronological),
                    (
                        f"Fenêtre {position}/{len(chronological)} · "
                        f"profil arrêté au {window.train_end}"
                    ),
                )
            targets, missing_mapping = _target_rows(items, window)
            snapshot = root / f"{window.window_id}.db"
            train_ids = {
                str(items[index]["id"]) for index in window.train_indexes
            }
            _training_snapshot(original, snapshot, train_ids)
            profile = build_profile(
                snapshot,
                train_personal_model=False,
                persist=False,
            )
            start_date = _period_start(window.test_end, period_years)
            candidates, source_counts, diagnostics = build_candidate_pool(
                client,
                profile,
                snapshot,
                start_date=start_date,
                end_date=window.test_end,
                settings=SEARCH_DEPTHS[depth],
                reliability=reliability,
                excluded_genre_ids=None,
                excluded_genre=lambda _candidate, _excluded: False,
                trace_ids={int(target["tmdb_id"]) for target in targets},
            )
            candidates = _exclude_training_history(
                candidates,
                snapshot,
                diagnostics,
            )
            metrics = evaluate_vivier_pool(
                candidates,
                targets,
                budgets=budgets,
            )
            eligible_targets = [
                target
                for target in targets
                if _within_period(target, start_date, window.test_end)
            ]
            reports.append(
                {
                    "manifest": window_manifest(
                        items,
                        window,
                        arena_version=VIVIER_AUDIT_VERSION,
                    ),
                    "search_window": {
                        "release_start": start_date,
                        "release_end": window.test_end,
                        "period_years": period_years,
                    },
                    "liked_without_tmdb_id": missing_mapping,
                    "source_counts": source_counts,
                    "pool_diagnostics": diagnostics,
                    "metrics": metrics,
                    "eligible_period_metrics": _retrieval_metrics(
                        candidates,
                        eligible_targets,
                        budgets=budgets,
                    ),
                    "missing_traces": _missing_traces(
                        targets,
                        diagnostics,
                        budgets=budgets,
                    ),
                }
            )

        fingerprint_after = _database_fingerprint(original)
        unchanged = fingerprint_before == fingerprint_after
        if not unchanged:
            raise RuntimeError(
                "La base a changé pendant l’audit du vivier. Relance-le sans "
                "import ni enrichissement en parallèle."
            )

    if on_progress:
        on_progress(len(chronological), len(chronological), "Audit terminé")
    summary = _aggregate_windows(reports, budgets=budgets)
    payload = {
        "schema_version": VIVIER_AUDIT_SCHEMA_VERSION,
        "audit_version": VIVIER_AUDIT_VERSION,
        "app_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": (
            "Mesurer si le vivier actuel retrouve les films notés 8+ dans "
            "des périodes futures, avant tout changement de source ou de règle."
        ),
        "methodology": {
            "protocol": "fenêtres chronologiques expansives et disjointes",
            "positive_definition": f"note IMDb ≥ {LIKE_THRESHOLD:g}/10",
            "budgets": list(budgets),
            "depth": depth,
            "reliability": reliability,
            "release_period_years": period_years,
            "ranking_excluded": True,
            "genre_exclusions_disabled": True,
            "source_ablation": (
                "chaque source est retirée des candidats déjà collectés, "
                "puis l’ordre équilibré est reconstruit"
            ),
            "historical_limit": (
                "les notes sont coupées dans le temps, mais TMDB fournit ses "
                "métadonnées, votes et pages de popularité actuels"
            ),
        },
        "window_metadata": window_metadata,
        "summary": summary,
        "windows": reports,
        "integrity": {
            "source_database_snapshot_used": True,
            "source_fingerprint_before": fingerprint_before,
            "source_fingerprint_after": fingerprint_after,
            "source_unchanged": True,
        },
        "run_id": stable_hash(
            {
                "audit_version": VIVIER_AUDIT_VERSION,
                "windows": [
                    report["manifest"]["split_hash"] for report in reports
                ],
                "settings": {
                    "depth": depth,
                    "reliability": reliability,
                    "period_years": period_years,
                    "budgets": budgets,
                },
            }
        )[:20],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    save_vivier_audit_report(original, payload)
    logger.info(
        "vivier_audit_completed | windows=%s | recall500=%s | elapsed=%s",
        len(reports),
        summary.get("recall_at_500"),
        payload["elapsed_seconds"],
    )
    gc.collect()
    return payload


def vivier_audit_report_path(
    database: str | Path | None,
    payload: dict,
) -> Path:
    target = Path(database or "data/cineprofile.db").parent / "logs"
    target.mkdir(parents=True, exist_ok=True)
    created = datetime.fromisoformat(str(payload["created_at"]))
    stamp = created.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return target / f"vivier_audit_{stamp}.json"


def save_vivier_audit_report(
    database: str | Path | None,
    payload: dict,
) -> Path:
    path = vivier_audit_report_path(database, payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def latest_vivier_audit_report(
    database: str | Path | None,
) -> Path | None:
    directory = Path(database or "data/cineprofile.db").parent / "logs"
    reports = sorted(directory.glob("vivier_audit_*.json"), reverse=True)
    return reports[0] if reports else None
