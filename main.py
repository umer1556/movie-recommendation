"""
main.py
-------
Streamlit entry point for the Movie Recommendation System.

Run locally:
    streamlit run main.py

The app has three logical sections rendered on one page:
  1. Sidebar  – app branding, user-preference panel, CF user selector
  2. Home     – search bar + favourite-movies multi-select
  3. Results  – recommendation cards (poster + metadata)
"""

import sys
import os


import streamlit as st
import pandas as pd

from src.data_loader import load_movies, load_ratings, build_similarity_matrix, build_user_movie_matrix
from src.recommender  import (
    get_recommendations_by_movie,
    get_recommendations_by_titles,
    get_recommendations_for_user,
    search_movies,
)
from src.tmdb_client import get_poster_url, get_tmdb_metadata, api_key_configured


# ──────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="CineMatch – Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Custom CSS  (minimal – keeps it readable)
# ──────────────────────────────────────────────

st.markdown(
    """
    <style>
        /* Card container */
        .movie-card {
            background: #1e1e2e;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 8px;
            border: 1px solid #2a2a3e;
            transition: border-color 0.2s;
        }
        .movie-card:hover { border-color: #e50914; }

        /* Score badge */
        .score-badge {
            display: inline-block;
            background: #e50914;
            color: white;
            padding: 2px 8px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        /* Genre tag */
        .genre-tag {
            display: inline-block;
            background: #2a2a3e;
            color: #a0a0c0;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.72rem;
            margin: 2px;
        }

        /* Section headers */
        .section-header {
            font-size: 1.4rem;
            font-weight: 700;
            color: #ffffff;
            border-left: 4px solid #e50914;
            padding-left: 10px;
            margin: 20px 0 12px 0;
        }

        /* Sidebar branding */
        .sidebar-brand {
            font-size: 1.6rem;
            font-weight: 800;
            color: #e50914;
            letter-spacing: -0.5px;
        }

        /* Hide Streamlit default footer */
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Data loading  (cached – runs once per session)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading movie database…")
def get_data():
    """Load all data and pre-compute similarity matrix."""
    movies_df         = load_movies()
    ratings_df        = load_ratings()
    similarity_matrix = build_similarity_matrix(movies_df)
    user_movie_matrix = build_user_movie_matrix(movies_df, ratings_df)
    return movies_df, ratings_df, similarity_matrix, user_movie_matrix


try:
    movies_df, ratings_df, sim_matrix, user_movie_matrix = get_data()
except FileNotFoundError as e:
    st.error(f"**Data error:** {e}")
    st.stop()


ALL_TITLES = movies_df["title"].sort_values().tolist()


# ──────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────

defaults = {
    "favourites":       [],       # list of movie titles the user likes
    "last_search":      "",       # last typed search query
    "recommendations":  None,     # cached recommendations DataFrame
    "rec_mode":         "search", # 'search' | 'favourites' | 'cf'
    "selected_user":    None,     # user_id for CF mode
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🎬 CineMatch</div>', unsafe_allow_html=True)
    st.caption("Personalised movie recommendations powered by content similarity.")
    st.divider()

    # ── TMDB status ──
    if api_key_configured():
        st.success("🖼 TMDB posters enabled", icon="✅")
    else:
        st.info(
            "Add **TMDB_API_KEY** to `.streamlit/secrets.toml` to enable movie posters.",
            icon="ℹ️",
        )
    st.divider()

    # ── Recommendation mode ──
    st.markdown("#### Recommendation Mode")
    mode = st.radio(
        label="mode",
        options=["🔍 Search by title", "❤️ From my favourites", "👥 Collaborative (CF)"],
        label_visibility="collapsed",
    )
    st.session_state["rec_mode"] = (
        "search"     if "Search"  in mode else
        "favourites" if "favour"  in mode else
        "cf"
    )

    st.divider()

    # ── Favourites management ──
    st.markdown("#### My Favourites")
    new_fav = st.selectbox(
        "Add a movie to your list",
        options=[""] + [t for t in ALL_TITLES if t not in st.session_state["favourites"]],
        key="fav_picker",
    )
    if new_fav:
        st.session_state["favourites"].append(new_fav)
        st.rerun()

    if st.session_state["favourites"]:
        st.write("Your list:")
        for i, fav in enumerate(st.session_state["favourites"]):
            col_fav, col_rm = st.columns([4, 1])
            col_fav.markdown(f"🎥 {fav}")
            if col_rm.button("✕", key=f"rm_{i}", help=f"Remove {fav}"):
                st.session_state["favourites"].pop(i)
                st.rerun()

        if st.button("Clear all favourites", use_container_width=True):
            st.session_state["favourites"] = []
            st.rerun()
    else:
        st.caption("No favourites yet. Add titles above.")

    st.divider()

    # ── Number of recommendations ──
    n_recs = st.slider("Recommendations to show", min_value=3, max_value=15, value=6)


# ──────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────

st.markdown("## 🎬 CineMatch")
st.markdown("Find movies you'll love based on what you already enjoy.")
st.divider()

rec_mode = st.session_state["rec_mode"]

# ── Search Mode ──
if rec_mode == "search":
    st.markdown('<div class="section-header">Search by Movie Title</div>', unsafe_allow_html=True)

    search_query = st.text_input(
        label="Type a movie title",
        placeholder="e.g. Inception, The Dark Knight…",
        value=st.session_state["last_search"],
    )

    col_search, col_clear = st.columns([2, 1])

    with col_search:
        run_search = st.button("Get Recommendations", type="primary", use_container_width=True)

    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state["last_search"] = ""
            st.session_state["recommendations"] = None
            st.rerun()

    if run_search and search_query:
        st.session_state["last_search"] = search_query
        with st.spinner("Finding similar movies…"):
            try:
                recs = get_recommendations_by_movie(
                    title=search_query,
                    movies_df=movies_df,
                    similarity_matrix=sim_matrix,
                    n=n_recs,
                )
                st.session_state["recommendations"] = recs
                st.session_state["rec_mode_used"] = "cbf_single"
            except ValueError as e:
                st.warning(f"⚠️ {e}")

                # Suggest close matches
                matches = search_movies(search_query, movies_df, max_results=5)
                if not matches.empty:
                    st.markdown("**Did you mean one of these?**")
                    for _, row in matches.iterrows():
                        if st.button(f"📽 {row['title']} ({row['year']})", key=f"suggest_{row['title']}"):
                            st.session_state["last_search"] = row["title"]
                            st.rerun()

# ── Favourites Mode ──
elif rec_mode == "favourites":
    st.markdown('<div class="section-header">Recommendations from Your Favourites</div>', unsafe_allow_html=True)

    if not st.session_state["favourites"]:
        st.info("Add movies to your favourites list in the sidebar first.", icon="👈")
    else:
        st.markdown(f"**Based on:** {', '.join(st.session_state['favourites'])}")
        with st.spinner("Blending your taste profile…"):
            try:
                recs = get_recommendations_by_titles(
                    titles=st.session_state["favourites"],
                    movies_df=movies_df,
                    similarity_matrix=sim_matrix,
                    n=n_recs,
                )
                st.session_state["recommendations"] = recs
                st.session_state["rec_mode_used"] = "cbf_multi"
            except ValueError as e:
                st.warning(str(e))

# ── Collaborative Filtering Mode ──
elif rec_mode == "cf":
    st.markdown('<div class="section-header">Collaborative Filtering (User-Based)</div>', unsafe_allow_html=True)

    if user_movie_matrix.empty:
        st.warning(
            "ratings.csv not found or is empty. "
            "Collaborative filtering is unavailable — switch to Search or Favourites mode.",
            icon="⚠️",
        )
    else:
        available_users = sorted(user_movie_matrix.index.tolist())
        selected_user = st.selectbox(
            "Select a user profile to simulate",
            options=available_users,
            index=0,
        )
        st.session_state["selected_user"] = selected_user

        if st.button("Get CF Recommendations", type="primary"):
            with st.spinner("Analysing similar users…"):
                recs = get_recommendations_for_user(
                    user_id=selected_user,
                    user_movie_matrix=user_movie_matrix,
                    movies_df=movies_df,
                    similarity_matrix=sim_matrix,
                    n=n_recs,
                )
                st.session_state["recommendations"] = recs
                st.session_state["rec_mode_used"] = "cf"

        # Show what this simulated user has already rated
        with st.expander("📊 This user's ratings"):
            user_rated = user_movie_matrix.loc[selected_user].dropna().sort_values(ascending=False)
            st.dataframe(
                user_rated.reset_index().rename(columns={"index": "Movie", selected_user: "Rating"}),
                use_container_width=True,
                hide_index=True,
            )


# ──────────────────────────────────────────────
# Results rendering
# ──────────────────────────────────────────────

recs: pd.DataFrame | None = st.session_state.get("recommendations")

if recs is not None and not recs.empty:
    mode_used = st.session_state.get("rec_mode_used", "cbf_single")
    label_map = {
        "cbf_single": "Content Similarity",
        "cbf_multi":  "Blended Taste Score",
        "cf":         "Predicted Rating",
    }
    score_col  = "similarity_score" if mode_used != "cf" else "predicted_rating"
    score_label = label_map.get(mode_used, "Score")

    st.markdown(f'<div class="section-header">Recommendations — {score_label}</div>', unsafe_allow_html=True)

    # Render cards in a 3-column grid
    cols = st.columns(3)

    for i, (_, row) in enumerate(recs.iterrows()):
        col = cols[i % 3]
        with col:
            # Fetch poster (fast – cached by lru_cache)
            if api_key_configured():
                meta = get_tmdb_metadata(row["title"], int(row.get("year", 0)))
                poster_url   = meta["poster_url"]
                vote_average = meta["vote_average"]
            else:
                poster_url   = None
                vote_average = 0.0

            with st.container():
                # Poster
                if poster_url:
                    st.image(poster_url, use_column_width=True)

                # Title + year
                st.markdown(f"**{row['title']}** ({int(row['year'])})")

                # Genre tags
                genres = row.get("genre", "").split()
                genre_html = " ".join(
                    f'<span class="genre-tag">{g.strip()}</span>'
                    for g in row.get("genre", "").replace(",", " ").split()
                    if g.strip()
                )
                st.markdown(genre_html, unsafe_allow_html=True)

                # Score badge
                score_val = row.get(score_col, 0.0)
                st.markdown(
                    f'<span class="score-badge">{score_label}: {score_val:.2f}</span>',
                    unsafe_allow_html=True,
                )

                if vote_average > 0:
                    st.markdown(
                        f'<span class="score-badge" style="background:#f5c518; color:#000">⭐ {vote_average:.1f}</span>',
                        unsafe_allow_html=True,
                    )

                # Overview (collapsed to save space)
                with st.expander("Overview"):
                    st.caption(row.get("overview", "No overview available."))

                # Quick add to favourites
                if row["title"] not in st.session_state["favourites"]:
                    if st.button(f"+ Add to Favourites", key=f"add_fav_{i}"):
                        st.session_state["favourites"].append(row["title"])
                        st.toast(f"Added '{row['title']}' to favourites!", icon="❤️")
                        st.rerun()

                st.markdown("---")

elif recs is not None and recs.empty:
    st.info("No recommendations found. Try a different movie or adjust your favourites.", icon="🤔")


# ──────────────────────────────────────────────
# Debug panel (hidden by default)
# ──────────────────────────────────────────────

with st.expander("🔧 Debug – Raw Data Inspector", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Movies", "Ratings", "Session State"])

    with tab1:
        st.dataframe(movies_df[["movie_id", "title", "year", "genre"]], use_container_width=True)

    with tab2:
        if not ratings_df.empty:
            st.dataframe(ratings_df, use_container_width=True)
        else:
            st.caption("No ratings data loaded.")

    with tab3:
        # Show session state minus the large DataFrames
        safe_state = {
            k: v for k, v in st.session_state.items()
            if k != "recommendations"
        }
        st.json(safe_state)
        if st.session_state.get("recommendations") is not None:
            st.dataframe(st.session_state["recommendations"], use_container_width=True)
