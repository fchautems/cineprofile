from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Iterable

import pandas as pd
from jinja2 import BaseLoader, Environment

from .db import connect, initialize, transaction
from .personal_model import ensure_personal_model
from .public_rating import best_public_rating


MODEL_VERSION = "cineprofile-affinity-0.9.0"


def profile_needs_refresh(profile: dict | None, counts: dict[str, int]) -> bool:
    if profile is None or profile.get("model_version") != MODEL_VERSION:
        return True
    summary = profile.get("summary", {})
    return (
        int(summary.get("rated_titles", -1)) != int(counts["total"])
        or int(summary.get("enriched_titles", -1)) != int(counts["enriched"])
    )


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _affinity_rows(
    memberships: Iterable[tuple[str, str, float, float | None]],
    baseline: float,
    *,
    shrinkage: float,
    minimum: int = 1,
) -> list[dict]:
    buckets: dict[str, list[tuple[str, float, float | None]]] = defaultdict(list)
    for entity_id, name, user_rating, benchmark in memberships:
        buckets[f"{entity_id}\u241f{name}"].append(
            (name, float(user_rating), _finite(benchmark))
        )

    rows = []
    for key, observations in buckets.items():
        if len(observations) < minimum:
            continue
        entity_id, name = key.split("\u241f", 1)
        user_values = [item[1] for item in observations]
        residuals = [
            item[1] - item[2] for item in observations if item[2] is not None
        ]
        support = len(observations)
        confidence = support / (support + shrinkage)
        lift = mean(user_values) - baseline
        residual = mean(residuals) if residuals else None
        # La préférence personnelle domine ; le résiduel permet de détecter ce
        # que l'utilisateur aime plus que le public.
        raw_signal = lift + (0.45 * residual if residual is not None else 0.0)
        score = confidence * raw_signal
        rows.append(
            {
                "id": int(entity_id) if entity_id.isdigit() else entity_id,
                "name": name,
                "seen": support,
                "average_user_rating": round(mean(user_values), 3),
                "average_residual_vs_public": (
                    round(residual, 3) if residual is not None else None
                ),
                "confidence": round(confidence, 3),
                "affinity": round(score, 3),
            }
        )
    return sorted(rows, key=lambda row: (row["affinity"], row["seen"]), reverse=True)


def _rating_distribution(ratings: pd.Series) -> list[dict]:
    counts = ratings.round().astype(int).value_counts().sort_index()
    total = int(counts.sum())
    return [
        {
            "rating": int(rating),
            "count": int(count),
            "share": round(float(count) / total, 4),
        }
        for rating, count in counts.items()
    ]


def _json_names(raw_values: Iterable[str | None], field: str = "name") -> list[str]:
    names: list[str] = []
    for raw in raw_values:
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for item in payload:
            value = item.get(field) if isinstance(item, dict) else None
            if value:
                names.append(str(value))
    return names


def _simple_group(
    frame: pd.DataFrame,
    column: str,
    baseline: float,
    *,
    minimum: int = 1,
) -> list[dict]:
    work = frame.dropna(subset=[column]).copy()
    if work.empty:
        return []
    rows = []
    for key, group in work.groupby(column):
        if len(group) < minimum:
            continue
        support = len(group)
        confidence = support / (support + 4)
        average = float(group["user_rating"].mean())
        rows.append(
            {
                "name": str(key),
                "seen": support,
                "average_user_rating": round(average, 3),
                "affinity": round(confidence * (average - baseline), 3),
            }
        )
    return sorted(rows, key=lambda row: (row["affinity"], row["seen"]), reverse=True)


def _load_memberships(
    connection,
    benchmarks: dict[str, float | None],
    role: str | None = None,
) -> list[tuple]:
    if role:
        return [
            (
                row["entity_id"],
                row["name"],
                row["user_rating"],
                benchmarks.get(row["imdb_id"]),
            )
            for row in connection.execute(
                """
                SELECT t.imdb_id,
                       CAST(p.tmdb_id AS TEXT) AS entity_id,
                       p.name, t.user_rating
                FROM credits c
                JOIN people p ON p.tmdb_id=c.person_id
                JOIN titles t ON t.imdb_id=c.imdb_id
                WHERE c.role=?
                """,
                (role,),
            )
        ]
    return []


def build_profile(database: str | Path | None = None) -> dict:
    initialize(database)
    with connect(database) as connection:
        titles = pd.read_sql_query("SELECT * FROM titles", connection)
        if titles.empty:
            raise ValueError("La base ne contient encore aucune évaluation IMDb.")
        titles["user_rating"] = pd.to_numeric(titles["user_rating"], errors="coerce")
        titles["imdb_rating"] = pd.to_numeric(titles["imdb_rating"], errors="coerce")
        titles["tmdb_rating"] = pd.to_numeric(titles["tmdb_rating"], errors="coerce")
        titles["benchmark"] = titles.apply(
            lambda row: best_public_rating(
                tmdb_rating=row.get("tmdb_rating"),
                tmdb_votes=row.get("tmdb_vote_count"),
                imdb_rating=row.get("imdb_rating"),
                imdb_votes=row.get("num_votes"),
            ).adjusted_rating,
            axis=1,
        )
        benchmark_by_imdb = {
            str(row.imdb_id): _finite(row.benchmark)
            for row in titles.itertuples()
        }
        baseline = float(titles["user_rating"].mean())

        genre_memberships = [
            (
                row["entity_id"],
                row["name"],
                row["user_rating"],
                benchmark_by_imdb.get(row["imdb_id"]),
            )
            for row in connection.execute(
                """
                SELECT t.imdb_id,
                       CAST(g.tmdb_id AS TEXT) AS entity_id,
                       g.name, t.user_rating
                FROM title_genres tg
                JOIN genres g ON g.tmdb_id=tg.genre_id
                JOIN titles t ON t.imdb_id=tg.imdb_id
                """
            )
        ]
        keyword_memberships = [
            (
                row["entity_id"],
                row["name"],
                row["user_rating"],
                benchmark_by_imdb.get(row["imdb_id"]),
            )
            for row in connection.execute(
                """
                SELECT t.imdb_id,
                       CAST(k.tmdb_id AS TEXT) AS entity_id,
                       k.name, t.user_rating
                FROM title_keywords tk
                JOIN keywords k ON k.tmdb_id=tk.keyword_id
                JOIN titles t ON t.imdb_id=tk.imdb_id
                """
            )
        ]
        role_rows = {
            role: _load_memberships(connection, benchmark_by_imdb, role)
            for role in (
                "director",
                "writer",
                "cast",
                "cinematography",
                "composer",
                "editor",
            )
        }
        genre_covered_titles = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT imdb_id FROM title_genres"
            )
        }
        director_covered_titles = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT imdb_id FROM credits WHERE role='director'"
            )
        }

        provider_counts = [
            dict(row)
            for row in connection.execute(
                """
                SELECT provider_name AS name, access_type, COUNT(*) AS titles
                FROM providers
                GROUP BY provider_name, access_type
                ORDER BY titles DESC, provider_name
                """
            )
        ]

    # Repli sur les genres du CSV avant que l'enrichissement ne soit terminé.
    for row in titles.itertuples():
        if row.imdb_id not in genre_covered_titles:
            for name in str(row.genres_csv or "").split(","):
                name = name.strip()
                if name:
                    genre_memberships.append(
                        (
                            f"csv:{name.casefold()}",
                            name,
                            row.user_rating,
                            row.benchmark,
                        )
                    )

    # Repli semblable pour les réalisateurs fournis par certains exports IMDb.
    for row in titles.itertuples():
        if row.imdb_id not in director_covered_titles:
            for name in str(row.directors_csv or "").split(","):
                name = name.strip()
                if name:
                    role_rows["director"].append(
                        (
                            f"csv:{name.casefold()}",
                            name,
                            row.user_rating,
                            row.benchmark,
                        )
                    )

    years = pd.to_numeric(titles["year"], errors="coerce")
    titles["decade"] = (years // 10 * 10).astype("Int64").astype(str).replace("<NA>", None)
    titles["runtime_band"] = pd.cut(
        pd.to_numeric(titles["runtime_minutes"], errors="coerce"),
        bins=[0, 89, 119, 149, 10000],
        labels=["< 90 min", "90–119 min", "120–149 min", "150 min et +"],
    ).astype(object)

    languages = _simple_group(titles, "original_language", baseline, minimum=2)
    decades = _simple_group(titles, "decade", baseline, minimum=2)
    runtimes = _simple_group(titles, "runtime_band", baseline, minimum=2)

    countries = Counter(_json_names(titles["countries_json"]))
    companies = Counter(_json_names(titles["companies_json"]))
    valid_benchmarks = titles.dropna(subset=["benchmark"])
    residual = (
        float((valid_benchmarks["user_rating"] - valid_benchmarks["benchmark"]).mean())
        if not valid_benchmarks.empty
        else None
    )

    profile = {
        "schema_version": "1.0",
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "methodology": {
            "baseline": "moyenne des notes personnelles",
            "affinity": (
                "écart à la moyenne personnelle, corrigé par l’écart à la note "
                "publique et réduit lorsque le nombre d’observations est faible"
            ),
            "interpretation": (
                "une affinité positive est un signal personnel, pas une prédiction "
                "certaine ; le champ confidence augmente avec le nombre de films vus"
            ),
            "recommendation_weights": {
                "genres": 0.10,
                "themes": 0.20,
                "people": 0.15,
                "semantic_neighbors": 0.40,
                "explicit_preferences": 0.15,
            },
            "exposure_guard": (
                "la fréquence d’un genre augmente la confiance mais ne constitue "
                "jamais à elle seule une préférence"
            ),
        },
        "summary": {
            "rated_titles": int(len(titles)),
            "enriched_titles": int((titles["metadata_status"] == "done").sum()),
            "enrichment_rate": round(
                float((titles["metadata_status"] == "done").mean()), 4
            ),
            "average_rating": round(baseline, 3),
            "median_rating": round(float(titles["user_rating"].median()), 3),
            "rating_stddev": round(float(titles["user_rating"].std(ddof=0)), 3),
            "share_8_or_more": round(float((titles["user_rating"] >= 8).mean()), 4),
            "share_5_or_less": round(float((titles["user_rating"] <= 5).mean()), 4),
            "average_residual_vs_public": round(residual, 3) if residual else residual,
            "first_rating_date": (
                str(titles["date_rated"].replace("", pd.NA).dropna().min())
                if titles["date_rated"].replace("", pd.NA).notna().any()
                else None
            ),
            "last_rating_date": (
                str(titles["date_rated"].replace("", pd.NA).dropna().max())
                if titles["date_rated"].replace("", pd.NA).notna().any()
                else None
            ),
        },
        "rating_distribution": _rating_distribution(titles["user_rating"]),
        "dimensions": {
            "genres": _affinity_rows(
                genre_memberships, baseline, shrinkage=8, minimum=2
            ),
            "directors": _affinity_rows(
                role_rows["director"], baseline, shrinkage=3, minimum=2
            ),
            "writers": _affinity_rows(
                role_rows["writer"], baseline, shrinkage=3, minimum=2
            ),
            "actors": _affinity_rows(
                role_rows["cast"], baseline, shrinkage=6, minimum=5
            ),
            "cinematographers": _affinity_rows(
                role_rows["cinematography"], baseline, shrinkage=4, minimum=3
            ),
            "composers": _affinity_rows(
                role_rows["composer"], baseline, shrinkage=4, minimum=3
            ),
            "editors": _affinity_rows(
                role_rows["editor"], baseline, shrinkage=4, minimum=3
            ),
            "keywords": _affinity_rows(
                keyword_memberships, baseline, shrinkage=5, minimum=2
            ),
            "languages": languages,
            "decades": decades,
            "runtime_bands": runtimes,
            "countries_by_exposure": [
                {"name": name, "seen": count} for name, count in countries.most_common(30)
            ],
            "production_companies_by_exposure": [
                {"name": name, "seen": count} for name, count in companies.most_common(30)
            ],
            "watch_providers": provider_counts,
        },
    }

    personal_model = ensure_personal_model(database)
    profile["personal_model"] = personal_model.summary
    profile["methodology"]["personal_prediction"] = (
        "les notes existantes sont masquées par groupes, prédites sans avoir "
        "été vues pendant l’apprentissage, puis utilisées pour calibrer la "
        "probabilité d’attribuer au moins 8/10"
    )

    with transaction(database) as connection:
        cursor = connection.execute(
            """
            INSERT INTO profile_runs(
              created_at, model_version, rated_count, enriched_count, profile_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                profile["generated_at"],
                MODEL_VERSION,
                profile["summary"]["rated_titles"],
                profile["summary"]["enriched_titles"],
                json.dumps(profile, ensure_ascii=False),
            ),
        )
        profile["profile_run_id"] = cursor.lastrowid
    return profile


REPORT_TEMPLATE = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Mon profil cinématographique</title>
  <style>
    :root { --ink:#171714; --muted:#6f7068; --paper:#f6f4ed;
      --card:#fffefa; --accent:#d24b33; --line:#ddd9cc; --good:#236b4b; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--paper); color:var(--ink);
      font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif; }
    main { max-width:1120px; margin:auto; padding:56px 24px 80px; }
    h1 { margin:0; font-family:Georgia,serif; font-size:clamp(2.6rem,7vw,5.8rem);
      line-height:.92; letter-spacing:-.055em; max-width:900px; }
    .lede { max-width:740px; color:var(--muted); font-size:1.1rem;
      line-height:1.6; margin:26px 0 36px; }
    .metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
    .metric,.panel { background:var(--card); border:1px solid var(--line);
      border-radius:18px; padding:20px; }
    .metric strong { display:block; font-size:2rem; }
    .metric span { color:var(--muted); font-size:.88rem; }
    section { margin-top:52px; }
    h2 { font-family:Georgia,serif; font-size:2.1rem; margin:0 0 8px; }
    .hint { color:var(--muted); margin:0 0 18px; }
    .grid { display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }
    table { width:100%; border-collapse:collapse; font-size:.92rem; }
    th,td { padding:10px 8px; border-bottom:1px solid var(--line); text-align:left; }
    th { color:var(--muted); font-weight:600; }
    td:nth-child(n+2),th:nth-child(n+2) { text-align:right; }
    .positive { color:var(--good); font-weight:700; }
    .negative { color:var(--accent); font-weight:700; }
    footer { color:var(--muted); border-top:1px solid var(--line);
      margin-top:52px; padding-top:20px; font-size:.84rem; }
    @media(max-width:760px) { .metrics,.grid { grid-template-columns:1fr 1fr; } }
    @media(max-width:500px) { .metrics,.grid { grid-template-columns:1fr; } }
  </style>
</head>
<body><main>
  <h1>Ton empreinte cinématographique</h1>
  <p class="lede">Ce rapport décrit non seulement ce que tu regardes, mais ce
  que tu apprécies davantage — ou moins — que ton propre niveau moyen et que le
  public. Les affinités sont corrigées pour ne pas surinterpréter une rencontre
  unique avec un acteur ou un réalisateur.</p>
  <div class="metrics">
    <div class="metric"><strong>{{ summary.rated_titles }}</strong><span>titres notés</span></div>
    <div class="metric"><strong>{{ "%.2f"|format(summary.average_rating) }}</strong><span>note personnelle moyenne</span></div>
    <div class="metric"><strong>{{ "%.0f%%"|format(summary.enrichment_rate * 100) }}</strong><span>données enrichies</span></div>
    <div class="metric"><strong>{{ "%.0f%%"|format(summary.share_8_or_more) }}</strong><span>notes de 8 ou plus</span></div>
  </div>
  {% for left_key, left_title, right_key, right_title in sections %}
  <section>
    <div class="grid">
      {% for key, title in [(left_key,left_title),(right_key,right_title)] %}
      <div class="panel">
        <h2>{{ title }}</h2>
        <p class="hint">Signaux les plus caractéristiques, avec leur fiabilité.</p>
        <table><thead><tr><th>Nom</th><th>Vus</th><th>Moy.</th><th>Affinité</th></tr></thead>
        <tbody>
        {% for row in dimensions[key][:12] %}
          <tr><td>{{ row.name }}</td><td>{{ row.seen }}</td>
          <td>{{ "%.2f"|format(row.average_user_rating) }}</td>
          <td class="{{ 'positive' if row.affinity >= 0 else 'negative' }}">
            {{ "%+.2f"|format(row.affinity) }}</td></tr>
        {% else %}
          <tr><td colspan="4">Enrichissement nécessaire pour cette dimension.</td></tr>
        {% endfor %}
        </tbody></table>
      </div>
      {% endfor %}
    </div>
  </section>
  {% endfor %}
  <footer>Modèle {{ model_version }} · Généré le {{ generated_at }} · Les
  affinités sont des signaux statistiques explicables, pas des jugements
  absolus.</footer>
</main></body></html>
"""


def render_report(profile: dict) -> str:
    environment = Environment(
        loader=BaseLoader(),
        autoescape=True,
    )
    template = environment.from_string(REPORT_TEMPLATE)
    return template.render(
        **profile,
        sections=[
            ("genres", "Genres", "keywords", "Thèmes"),
            ("directors", "Réalisateurs", "writers", "Scénaristes"),
            ("actors", "Interprètes", "cinematographers", "Image"),
            ("composers", "Musique", "decades", "Décennies"),
            ("languages", "Langues", "runtime_bands", "Durées"),
        ],
    )
