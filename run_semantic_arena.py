from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cineprofile.semantic_arena import (
    semantic_report_path,
    run_semantic_arena,
)
from cineprofile.semantic_models import (
    SEMANTIC_MODELS,
    estimated_missing_download_gib,
)
from cineprofile.semantic import embedding_execution_providers
from cineprofile.diagnostics import configure_logging


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MiniLM, Multilingual E5 Large et BGE-M3 sans modifier "
            "CineProfile."
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
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    database = Path(arguments.database)
    logger = configure_logging(database)
    model_directory = database.parent / "models"
    missing_gib = estimated_missing_download_gib(model_directory)
    try:
        execution_providers = embedding_execution_providers()
    except Exception as exc:
        logger.exception(
            "semantic_arena_provider_initialization_failed | error=%s",
            exc,
        )
        print()
        print(f"[Laboratoire v1] Initialisation du calcul impossible : {exc}")
        return 1
    print()
    print("[Laboratoire v1] Étape 3 · deep learning sémantique")
    print(f"[Laboratoire v1] Base : {database.resolve()}")
    print(
        "[Laboratoire v1] Modèles : "
        + ", ".join(spec.label for spec in SEMANTIC_MODELS)
    )
    print(
        f"[Laboratoire v1] Téléchargement encore estimé : {missing_gib:.2f} Gio."
    )
    print(
        "[Laboratoire v1] Calcul : "
        + (
            "GPU NVIDIA (CUDA 12)"
            if execution_providers[0] == "CUDAExecutionProvider"
            else "CPU"
        )
    )
    print(
        "[Laboratoire v1] Les modèles et tous les calculs restent sur ce PC."
    )
    print(
        "[Laboratoire v1] CineProfile et sa base ne seront pas modifiés.",
        flush=True,
    )
    previous_message = ""

    def progress(current: int, total: int, message: str) -> None:
        nonlocal previous_message
        if message == previous_message:
            return
        previous_message = message
        if total > 0:
            percent = min(100, round(100 * current / total))
            print(f"[{percent:3d}%] {message}", flush=True)
        else:
            print(f"[ ... ] {message}", flush=True)

    try:
        logger.info(
            "semantic_arena_started | providers=%s",
            ",".join(execution_providers),
        )
        payload = run_semantic_arena(
            database,
            baseline_report=arguments.baseline,
            on_progress=progress,
        )
    except Exception as exc:
        logger.exception("semantic_arena_failed | error=%s", exc)
        print()
        print(f"[Laboratoire v1] Échec de l’étape 3 : {exc}")
        print("[Laboratoire v1] Détail : data\\logs\\cineprofile.log")
        return 1

    report = semantic_report_path(database, payload)
    summaries = payload["satisfaction_engine_summaries"]
    print()
    print("[Laboratoire v1] Étape 3 terminée.")
    for spec in SEMANTIC_MODELS:
        metrics = summaries[spec.engine]
        ndcg = metrics.get("ndcg_at_20", {}).get("mean")
        top = metrics.get("average_rating_at_20", {}).get("mean")
        print(
            f"[Laboratoire v1] {spec.label} : "
            f"NDCG@20={ndcg}, note moyenne top 20={top}"
        )
    decision = payload["promotion_decision"]
    winner = decision["best_semantic_engine"]
    print(f"[Laboratoire v1] Meilleur challenger : {winner}")
    print(
        "[Laboratoire v1] Décision automatique : "
        + (
            "passage au test de reranking profond."
            if decision["promote_to_reranking_test"]
            else "aucun passage automatique au reranking."
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
