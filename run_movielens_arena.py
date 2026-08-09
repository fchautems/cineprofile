from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cineprofile.movielens_arena import (
    movielens_report_path,
    run_movielens_arena,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Télécharge MovieLens 32M puis évalue le challenger "
            "collaboratif sans modifier CineProfile."
        )
    )
    parser.add_argument(
        "--database",
        default=os.getenv("CINEPROFILE_DB", "data/cineprofile.db"),
        help="Chemin de la base CineProfile.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Rapport étape 1 ; le plus récent est choisi par défaut.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Dossier MovieLens ; data/movielens par défaut.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    database = Path(arguments.database)
    print()
    print("[Laboratoire v1] Étape 2 · challenger MovieLens")
    print(f"[Laboratoire v1] Base : {database.resolve()}")
    print(
        "[Laboratoire v1] Le premier lancement télécharge environ 239 Mio "
        "et prépare 32 millions de notes."
    )
    print(
        "[Laboratoire v1] CineProfile et sa base ne seront pas modifiés.",
        flush=True,
    )
    previous_message = ""

    def progress(current: int, total: int, message: str) -> None:
        nonlocal previous_message
        if message == previous_message and total > 0:
            return
        previous_message = message
        if total > 0:
            percent = min(100, round(100 * current / total))
            print(f"[{percent:3d}%] {message}", flush=True)
        else:
            print(f"[ ... ] {message}", flush=True)

    try:
        payload = run_movielens_arena(
            database,
            baseline_report=arguments.baseline,
            dataset_root=arguments.dataset_root,
            on_progress=progress,
        )
    except Exception as exc:
        print()
        print(f"[Laboratoire v1] Échec de l’étape 2 : {exc}")
        return 1

    report = movielens_report_path(database, payload)
    summaries = payload["satisfaction_engine_summaries"]
    print()
    print("[Laboratoire v1] Étape 2 terminée.")
    for engine, metrics in summaries.items():
        ndcg = metrics.get("ndcg_at_20", {}).get("mean")
        top = metrics.get("average_rating_at_20", {}).get("mean")
        print(
            f"[Laboratoire v1] {engine} : "
            f"NDCG@20={ndcg}, note moyenne top 20={top}"
        )
    promoted = payload["promotion_decision"]["promote_to_v1_foundation"]
    print(
        "[Laboratoire v1] Décision automatique : "
        + (
            "MovieLens mérite de passer à l’étape hybride."
            if promoted
            else "MovieLens ne bat pas encore les références."
        )
    )
    print(
        "[Laboratoire v1] Base inchangée : "
        + ("oui" if payload["integrity"]["source_unchanged"] else "NON")
    )
    print(f"[Laboratoire v1] Rapport à m’envoyer : {report.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
