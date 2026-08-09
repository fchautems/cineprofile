from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import numpy as np

from . import personal_model as pm


DEFAULT_WINDOWS = 5
MINIMUM_ITEMS = 120
MINIMUM_TRAIN_ITEMS = 80
MINIMUM_TEST_ITEMS = 20


@dataclass(frozen=True)
class ChronologicalWindow:
    window_id: str
    train_indexes: tuple[int, ...]
    test_indexes: tuple[int, ...]
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def rated_date(item: dict) -> date | None:
    raw = str(item.get("date_rated") or "").strip()
    if len(raw) < 10:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _item_key(item: dict) -> str:
    return str(item.get("id") or item.get("imdb_id") or "")


def stable_hash(rows: object) -> str:
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dated_groups(
    items: list[dict],
) -> tuple[list[tuple[str, tuple[int, ...]]], int]:
    groups: dict[str, list[int]] = defaultdict(list)
    undated = 0
    for index, item in enumerate(items):
        item_date = rated_date(item)
        if item_date is None:
            undated += 1
            continue
        groups[item_date.isoformat()].append(index)
    ordered: list[tuple[str, tuple[int, ...]]] = []
    for item_date in sorted(groups):
        indexes = tuple(
            sorted(groups[item_date], key=lambda index: _item_key(items[index]))
        )
        ordered.append((item_date, indexes))
    return ordered, undated


def _group_counts(
    groups: list[tuple[str, tuple[int, ...]]],
) -> list[int]:
    return [len(indexes) for _, indexes in groups]


def _training_boundary(
    groups: list[tuple[str, tuple[int, ...]]],
    *,
    minimum_train: int,
    minimum_future: int,
) -> int:
    counts = _group_counts(groups)
    total = sum(counts)
    target = max(minimum_train, total // 2)
    cumulative = np.cumsum(counts)
    choices = [
        boundary
        for boundary in range(1, len(groups))
        if int(cumulative[boundary - 1]) >= minimum_train
        and total - int(cumulative[boundary - 1]) >= minimum_future
    ]
    if not choices:
        raise ValueError(
            "Les dates disponibles ne permettent pas de séparer un passé "
            "d’apprentissage et au moins deux périodes de test."
        )
    return min(
        choices,
        key=lambda boundary: (
            abs(int(cumulative[boundary - 1]) - target),
            boundary,
        ),
    )


def _weighted_group_blocks(
    groups: list[tuple[str, tuple[int, ...]]],
    *,
    requested_windows: int,
    minimum_test: int,
) -> list[list[tuple[str, tuple[int, ...]]]]:
    if not groups:
        return []
    counts = _group_counts(groups)
    total = sum(counts)
    windows = min(
        max(2, int(requested_windows)),
        len(groups),
        max(1, total // minimum_test),
    )
    if windows < 2:
        return []

    cumulative = np.cumsum(counts)
    cuts = [0]
    for window in range(1, windows):
        target = total * window / windows
        lower = cuts[-1] + 1
        upper = len(groups) - (windows - window)
        choices = range(lower, upper + 1)
        boundary = min(
            choices,
            key=lambda index: (
                abs(int(cumulative[index - 1]) - target),
                index,
            ),
        )
        cuts.append(boundary)
    cuts.append(len(groups))
    blocks = [groups[cuts[index] : cuts[index + 1]] for index in range(len(cuts) - 1)]

    while len(blocks) > 1:
        sizes = [sum(len(indexes) for _, indexes in block) for block in blocks]
        small = next(
            (index for index, size in enumerate(sizes) if size < minimum_test),
            None,
        )
        if small is None:
            break
        if small == 0:
            blocks[1] = blocks[0] + blocks[1]
            del blocks[0]
        else:
            blocks[small - 1].extend(blocks[small])
            del blocks[small]
    return blocks if len(blocks) >= 2 else []


def build_chronological_windows(
    items: list[dict],
    *,
    requested_windows: int = DEFAULT_WINDOWS,
    minimum_train: int = MINIMUM_TRAIN_ITEMS,
    minimum_test: int = MINIMUM_TEST_ITEMS,
) -> tuple[list[ChronologicalWindow], dict]:
    if len(items) < MINIMUM_ITEMS:
        raise ValueError(
            f"L’arène demande au moins {MINIMUM_ITEMS} films exploitables."
        )
    requested_windows = max(2, min(10, int(requested_windows)))
    minimum_train = max(60, int(minimum_train))
    minimum_test = max(10, int(minimum_test))
    groups, undated = _dated_groups(items)
    dated_count = sum(len(indexes) for _, indexes in groups)
    if dated_count < MINIMUM_ITEMS:
        raise ValueError(
            "L’arène demande au moins 120 films avec une date de notation "
            "valide pour garantir un test chronologique."
        )
    boundary = _training_boundary(
        groups,
        minimum_train=minimum_train,
        minimum_future=2 * minimum_test,
    )
    training_groups = list(groups[:boundary])
    test_blocks = _weighted_group_blocks(
        list(groups[boundary:]),
        requested_windows=requested_windows,
        minimum_test=minimum_test,
    )
    if len(test_blocks) < 2:
        raise ValueError(
            "Impossible de construire au moins deux périodes de test "
            "chronologiques suffisamment grandes."
        )

    windows: list[ChronologicalWindow] = []
    previous_test_indexes: set[int] = set()
    expanding_groups = list(training_groups)
    for position, test_groups in enumerate(test_blocks, start=1):
        train_indexes = tuple(
            index for _, indexes in expanding_groups for index in indexes
        )
        test_indexes = tuple(index for _, indexes in test_groups for index in indexes)
        if set(train_indexes) & set(test_indexes):
            raise RuntimeError("Chevauchement interne entre passé et test.")
        if previous_test_indexes & set(test_indexes):
            raise RuntimeError("Une note apparaît dans plusieurs tests.")
        previous_test_indexes.update(test_indexes)
        windows.append(
            ChronologicalWindow(
                window_id=f"chrono_{position:02d}",
                train_indexes=train_indexes,
                test_indexes=test_indexes,
                train_start=expanding_groups[0][0],
                train_end=expanding_groups[-1][0],
                test_start=test_groups[0][0],
                test_end=test_groups[-1][0],
            )
        )
        expanding_groups.extend(test_groups)

    metadata = {
        "requested_windows": requested_windows,
        "constructed_windows": len(windows),
        "dated_items": dated_count,
        "undated_items_excluded": undated,
        "same_day_entries_kept_together": True,
        "initial_train_count": len(windows[0].train_indexes),
        "tested_items_once": len(previous_test_indexes),
    }
    return windows, metadata


def window_manifest(
    items: list[dict],
    window: ChronologicalWindow,
    *,
    arena_version: str,
) -> dict:
    def rows(indexes: tuple[int, ...]) -> list[dict]:
        return [
            {
                "id": _item_key(items[index]),
                "rating": float(items[index]["rating"]),
                "date_rated": str(items[index].get("date_rated") or ""),
            }
            for index in indexes
        ]

    train_rows = rows(window.train_indexes)
    test_rows = rows(window.test_indexes)
    train_ratings = np.asarray(
        [float(items[index]["rating"]) for index in window.train_indexes],
        dtype=float,
    )
    test_ratings = np.asarray(
        [float(items[index]["rating"]) for index in window.test_indexes],
        dtype=float,
    )
    return {
        "window_id": window.window_id,
        "train": {
            "count": len(train_rows),
            "start": window.train_start,
            "end": window.train_end,
            "positive_count": int(np.sum(train_ratings >= pm.LIKE_THRESHOLD)),
            "mean_rating": round(float(np.mean(train_ratings)), 4),
            "manifest_hash": stable_hash(train_rows),
        },
        "test": {
            "count": len(test_rows),
            "start": window.test_start,
            "end": window.test_end,
            "positive_count": int(np.sum(test_ratings >= pm.LIKE_THRESHOLD)),
            "mean_rating": round(float(np.mean(test_ratings)), 4),
            "manifest_hash": stable_hash(test_rows),
        },
        "split_hash": stable_hash(
            {
                "arena_version": arena_version,
                "train": train_rows,
                "test": test_rows,
            }
        ),
    }
