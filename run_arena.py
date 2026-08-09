from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cineprofile.arena import (
    DEFAULT_WINDOWS,
    arena_report_path,
    run_offline_arena,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lance l’arène chronologique CineProfile sans modifier "
            "l’application ni sa base."
        )
    )
    parser.add_argument(
        "--database",
        default=os.getenv("CINEPROFILE_DB", "data/cineprofile.db"),
        help="Chemin de la base CineProfile.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=DEFAULT_WINDOWS,
        help="Nombre maximal de fenêtres chronologiques (défaut : 5).",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    database = Path(arguments.database)
    print()
    print("[Laboratoire v1] Arène chronologique hors ligne")
    print(f"[Laboratoire v1] Base : {database.resolve()}")
    print("[Laboratoire v1] La base source sera contrôlée avant et après le test.")
    print(
        "[Laboratoire v1] Les vecteurs manquants seront calculés dans une "
        "copie temporaire."
    )

    def progress(current: int, total: int, message: str) -> None:
        print(f"[{current:02d}/{total:02d}] {message}", flush=True)

    try:
        payload = run_offline_arena(
            database,
            requested_windows=arguments.windows,
            on_progress=progress,
        )
    except Exception as exc:
        print()
        print(f"[Laboratoire v1] Échec : {exc}")
        print("[Laboratoire v1] Consulte data\\logs\\cineprofile.log pour le détail.")
        return 1

    report = arena_report_path(database, payload)
    summaries = payload.get("engine_summaries", {})
    print()
    print(
        f"[Laboratoire v1] Terminé : {len(payload['windows'])} fenêtres chronologiques."
    )
    print(
        "[Laboratoire v1] Base inchangée : "
        + ("oui" if payload["integrity"]["source_unchanged"] else "NON")
    )
    for engine, metrics in summaries.items():
        ndcg = metrics.get("ndcg_at_20", {}).get("mean")
        top_rating = metrics.get("average_rating_at_20", {}).get("mean")
        print(
            f"[Laboratoire v1] {engine} : "
            f"NDCG@20={ndcg}, note moyenne top 20={top_rating}"
        )
    print(f"[Laboratoire v1] Rapport à m’envoyer : {report.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
