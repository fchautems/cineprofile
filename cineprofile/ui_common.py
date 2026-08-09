from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from cineprofile.db import connect


def latest_profile(database: str | Path) -> dict | None:
    with connect(database) as connection:
        row = connection.execute(
            "SELECT id, profile_json FROM profile_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    profile = json.loads(row["profile_json"])
    profile["profile_run_id"] = row["id"]
    return profile


def metric_row(counts: dict[str, int], profile: dict | None) -> None:
    columns = st.columns(4)
    columns[0].metric("Titres notés", f"{counts['total']:,}".replace(",", "’"))
    columns[1].metric("Titres enrichis", f"{counts['enriched']:,}".replace(",", "’"))
    rate = counts["enriched"] / counts["total"] if counts["total"] else 0
    columns[2].metric("Couverture", f"{rate:.0%}")
    columns[3].metric(
        "Note moyenne",
        f"{profile['summary']['average_rating']:.2f}" if profile else "—",
    )
