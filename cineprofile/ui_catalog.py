from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from cineprofile.db import connect


@st.cache_data(ttl=15)
def load_catalog(database: str) -> pd.DataFrame:
    with connect(database) as connection:
        return pd.read_sql_query(
            """
            SELECT
              t.imdb_id, t.title, t.original_title, t.title_type, t.year,
              t.user_rating, t.imdb_rating, t.runtime_minutes, t.date_rated,
              t.original_language, t.metadata_status, t.overview, t.genres_csv,
              t.directors_csv,
              GROUP_CONCAT(DISTINCT g.name) AS enriched_genres,
              GROUP_CONCAT(DISTINCT p.name) AS people_names
            FROM titles t
            LEFT JOIN title_genres tg ON tg.imdb_id=t.imdb_id
            LEFT JOIN genres g ON g.tmdb_id=tg.genre_id
            LEFT JOIN credits c ON c.imdb_id=t.imdb_id
            LEFT JOIN people p ON p.tmdb_id=c.person_id
            GROUP BY t.imdb_id
            ORDER BY t.date_rated DESC, t.title
            """,
            connection,
        )


def clear_catalog_cache() -> None:
    load_catalog.clear()


def title_details(database: str | Path, imdb_id: str) -> dict:
    with connect(database) as connection:
        title = connection.execute(
            "SELECT * FROM titles WHERE imdb_id=?",
            (imdb_id,),
        ).fetchone()
        people = connection.execute(
            """
            SELECT c.role, c.job, p.name, c.character_name
            FROM credits c JOIN people p ON p.tmdb_id=c.person_id
            WHERE c.imdb_id=?
            ORDER BY
              CASE c.role WHEN 'director' THEN 0 WHEN 'writer' THEN 1
              WHEN 'cast' THEN 2 ELSE 3 END,
              c.credit_order, p.name
            """,
            (imdb_id,),
        ).fetchall()
        keywords = connection.execute(
            """
            SELECT k.name FROM title_keywords tk
            JOIN keywords k ON k.tmdb_id=tk.keyword_id
            WHERE tk.imdb_id=? ORDER BY k.name
            """,
            (imdb_id,),
        ).fetchall()
    return {
        "title": dict(title) if title else {},
        "people": [dict(row) for row in people],
        "keywords": [row["name"] for row in keywords],
    }


def _prepare_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    prepared = catalog.copy()
    for column in ("year", "user_rating", "imdb_rating"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared["genres"] = prepared["enriched_genres"].fillna(
        prepared["genres_csv"].fillna("")
    )
    return prepared


def render_catalog_tab(database: str | Path) -> None:
    st.subheader("Explorer tout l’historique")
    catalog = load_catalog(str(database))
    if catalog.empty:
        st.info("Importe d’abord le CSV IMDb.")
        return

    catalog = _prepare_catalog(catalog)
    known_years = catalog["year"].dropna()
    has_known_years = not known_years.empty
    minimum_year = int(known_years.min()) if not known_years.empty else 1900
    maximum_year = int(known_years.max()) if not known_years.empty else minimum_year
    all_types = sorted(
        value for value in catalog["title_type"].dropna().unique() if value
    )
    all_genres = sorted(
        {
            genre.strip()
            for cell in catalog["genres"]
            for genre in str(cell).split(",")
            if genre.strip()
        }
    )

    with st.expander("Filtres du catalogue", expanded=True):
        search_col, type_col = st.columns([2, 1])
        title_search = search_col.text_input(
            "Titre ou personne",
            placeholder="Matrix, Nolan, Bong Joon Ho…",
        )
        selected_types = type_col.multiselect(
            "Types",
            all_types,
            default=all_types,
        )
        year_col, rating_col = st.columns(2)
        if not has_known_years:
            year_col.text_input(
                "Années",
                value="Années inconnues",
                disabled=True,
            )
            year_range = None
        elif minimum_year == maximum_year:
            year_col.text_input(
                "Années",
                value=str(minimum_year),
                disabled=True,
            )
            year_range = (minimum_year, maximum_year)
        else:
            year_range = year_col.slider(
                "Années",
                minimum_year,
                maximum_year,
                (minimum_year, maximum_year),
            )
        rating_range = rating_col.slider(
            "Ta note",
            1.0,
            10.0,
            (1.0, 10.0),
            step=0.5,
        )
        genre_col, status_col = st.columns(2)
        selected_genres = genre_col.multiselect("Genres", all_genres)
        enrichment_status = status_col.multiselect(
            "État des données",
            ["Enrichi", "À enrichir", "Erreur ou absent"],
            default=["Enrichi", "À enrichir", "Erreur ou absent"],
        )

    filtered = catalog.copy()
    if title_search:
        mask = (
            filtered["title"].fillna("").str.contains(
                title_search, case=False, regex=False
            )
            | filtered["original_title"].fillna("").str.contains(
                title_search, case=False, regex=False
            )
            | filtered["people_names"].fillna("").str.contains(
                title_search, case=False, regex=False
            )
            | filtered["directors_csv"].fillna("").str.contains(
                title_search, case=False, regex=False
            )
        )
        filtered = filtered[mask]
    if selected_types:
        filtered = filtered[filtered["title_type"].isin(selected_types)]
    filtered = filtered[
        filtered["user_rating"].between(rating_range[0], rating_range[1])
    ]
    if year_range is not None:
        filtered = filtered[
            filtered["year"].between(year_range[0], year_range[1])
        ]
    if selected_genres:
        selected_genres_set = set(selected_genres)
        filtered = filtered[
            filtered["genres"].apply(
                lambda cell: bool(
                    selected_genres_set.intersection(
                        value.strip()
                        for value in str(cell).split(",")
                        if value.strip()
                    )
                )
            )
        ]

    allowed_statuses: set[str] = set()
    if "Enrichi" in enrichment_status:
        allowed_statuses.add("done")
    if "À enrichir" in enrichment_status:
        allowed_statuses.add("pending")
    if "Erreur ou absent" in enrichment_status:
        allowed_statuses.update({"error", "not_found"})
    filtered = filtered[filtered["metadata_status"].isin(allowed_statuses)]

    st.caption(f"{len(filtered):,} titre(s)".replace(",", "’"))
    st.dataframe(
        filtered[
            [
                "title",
                "year",
                "title_type",
                "user_rating",
                "imdb_rating",
                "genres",
                "date_rated",
                "metadata_status",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "title": "Titre",
            "year": "Année",
            "title_type": "Type",
            "user_rating": "Ta note",
            "imdb_rating": "IMDb",
            "genres": "Genres",
            "date_rated": "Noté le",
            "metadata_status": "Données",
        },
    )

    if filtered.empty:
        return

    labels = {
        (
            f"{row.title} ({int(row.year) if pd.notna(row.year) else '—'})"
            f" · {row.user_rating:g}/10 · {row.imdb_id}"
        ): row.imdb_id
        for row in filtered.head(500).itertuples()
    }
    selected_label = st.selectbox("Ouvrir une fiche", ["—"] + list(labels))
    if selected_label == "—":
        return

    details = title_details(database, labels[selected_label])
    title = details["title"]
    st.markdown(f"### {title.get('title')}")
    if title.get("overview"):
        st.write(title["overview"])
    people_frame = pd.DataFrame(details["people"])
    if not people_frame.empty:
        st.dataframe(
            people_frame[["role", "name", "job", "character_name"]],
            width="stretch",
            hide_index=True,
        )
    if details["keywords"]:
        st.caption("Thèmes · " + " · ".join(details["keywords"]))
