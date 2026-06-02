"""
main.py (UPDATED)
-------
Streamlit entry point for the Movie Recommendation System.
Now with: Actor search, improved UI, fuzzy matching.

Run locally:
    streamlit run main.py
"""

import streamlit as st
import pandas as pd

from data_loader import load_movies, load_ratings, build_similarity_matrix, build_user_movie_matrix
from recommender  import (
    get_recommendations_by_movie,
    get_recommendations_by_titles,
    get_recommendations_for_user,
    search_by_actor,
    sort_movies,
    search_movies,
)
from tmdb_client import get_poster_url, get_tmdb_metadata, api_key_configured


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
# Modern CSS Styling
# ──────────────────────────────────────────────

st.markdown(
    """
    <style>
        /* Main background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        }

        /* Card styles */
        .movie-card {
            background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 12px;
            border: 1px solid #e94560;
            box-shadow: 0 8px 32px rgba(233, 69, 96, 0.1);
            transition: all 0.3s ease;
        }
        .movie-card:hover {
            border-color: #ff6b6b;
            box-shadow: 0 12px 48px rgba(233, 69, 96, 0.25);
            transform: translateY(-4px);
        }

        /* Score badges */
        .score-badge {
            display: inline-block;
            background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            margin-right: 8px;
            margin-bottom: 8px;
            box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3);
        }

        /* Genre tags */
        .genre-tag {
            display: inline-block;
            background: rgba(233, 69, 96, 0.15);
            color: #ff6b6b;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 4px 4px 4px 0;
            border: 1px solid #e94560;
        }

        /* Section headers */
        .section-header {
            font-size: 1.8rem;
            font-weight: 800;
            color: #ffffff;
            background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 24px 0 16px 0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Sidebar branding */
        .sidebar-brand {
            font-size: 2rem;
            font-weight: 900;
            background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 12px;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 700;
            padding: 10px 20px;
            box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3);
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            box-shadow: 0 8px 25px rgba(233, 69, 96, 0.5);
            transform: translateY(-2px);
        }

        /* Input fields */
        input, textarea {
            background-color: #1a1a2e !important;
            color: white !important;
            border: 1px solid #e94560 !important;
            border-radius: 8px !important;
        }
        input::placeholder {
            color: #999 !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] button {
            background-color: #16213e;
            color: #999;
            border-bottom: 2px solid transparent;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #ff6b6b;
            border-bottom: 2px solid #e94560;
        }

        /* Hide footer */
        footer { visibility: hidden; }
        
        /* Text colors */
        h1, h2, h3 { color: #ffffff; }
        p { color: #ccc; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="🎬 Loading movie database…")
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
# Session state
# ──────────────────────────────────────────────

defaults = {
    "favourites":       [],
    "last_search":      "",
    "recommendations":  None,
    "rec_mode":         "search",
    "selected_user":    None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🎬 CineMatch</div>', unsafe_allow_html=True)
    st.caption("✨ Discover movies you'll love")
    st.divider()

    # Recommendation mode
    st.markdown("#### 🎯 What would you like to do?")
    mode = st.radio(
        label="mode",
        options=["🔍 Search Movies", "🎭 Find by Actor", "❤️ My Favourites", "👥 Similar Users"],
        label_visibility="collapsed",
    )
    st.session_state["rec_mode"] = (
        "search"     if "Search"  in mode else
        "actor"      if "Actor"   in mode else
        "favourites" if "Favour"  in mode else
        "cf"
    )

    st.divider()

    # Favourites management
    st.markdown("#### 💕 My Favourites")
    new_fav = st.selectbox(
        "Add a movie",
        options=[""] + [t for t in ALL_TITLES if t not in st.session_state["favourites"]],
        key="fav_picker",
    )
    if new_fav:
        st.session_state["favourites"].append(new_fav)
        st.rerun()

    if st.session_state["favourites"]:
        st.write(f"**{len(st.session_state['favourites'])} in your list:**")
        for i, fav in enumerate(st.session_state["favourites"]):
            col_fav, col_rm = st.columns([4, 1])
            col_fav.markdown(f"🎥 *{fav}*")
            if col_rm.button("✕", key=f"rm_{i}"):
                st.session_state["favourites"].pop(i)
                st.rerun()

        if st.button("🗑 Clear all", use_container_width=True):
            st.session_state["favourites"] = []
            st.rerun()
    else:
        st.caption("No favourites yet")

    st.divider()
    n_recs = st.slider("📊 Results to show", 3, 15, 6)


# ──────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────

st.markdown('<div class="section-header">🎬 CineMatch</div>', unsafe_allow_html=True)
st.markdown("**Your AI-powered movie discovery engine** — find films based on what you love.")
st.divider()

rec_mode = st.session_state["rec_mode"]

# ── SEARCH MODE ──
if rec_mode == "search":
    st.markdown('<div class="section-header">🔍 Search by Movie Title</div>', unsafe_allow_html=True)

    search_query = st.text_input(
        label="Movie title",
        placeholder="e.g. Inception, The Dark Knight, Parasite…",
        value=st.session_state["last_search"],
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        run_search = st.button("🎯 Get Recommendations", type="primary", use_container_width=True)

    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state["last_search"] = ""
            st.session_state["recommendations"] = None
            st.rerun()

    if run_search and search_query:
        st.session_state["last_search"] = search_query
        with st.spinner("🔍 Analyzing similarity…"):
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
                matches = search_movies(search_query, movies_df, max_results=5)
                if not matches.empty:
                    st.markdown("**Try one of these:**")
                    for _, row in matches.iterrows():
                        if st.button(f"📽 {row['title']} ({row['year']})", key=f"suggest_{row['title']}"):
                            st.session_state["last_search"] = row["title"]
                            st.rerun()

# ── ACTOR SEARCH MODE [NEW] ──
elif rec_mode == "actor":
    st.markdown('<div class="section-header">🎭 Find Movies by Actor</div>', unsafe_allow_html=True)

    actor_query = st.text_input(
        label="Actor name",
        placeholder="e.g. Leonardo DiCaprio, Morgan Freeman…",
    )

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        run_actor_search = st.button("🎬 Find Movies", type="primary", use_container_width=True)

    with col2:
        sort_option = st.radio("Sort by", ["Title (A-Z)", "Year (Newest)"], horizontal=True)

    with col3:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state["recommendations"] = None
            st.rerun()

    if run_actor_search and actor_query:
        with st.spinner("🔍 Searching for actor…"):
            recs = search_by_actor(actor_query, movies_df)
            if recs.empty:
                st.warning(f"No movies found with actor '{actor_query}'")
            else:
                # Sort results
                if sort_option == "Title (A-Z)":
                    recs = recs.sort_values("title", ascending=True).reset_index(drop=True)

                st.session_state["recommendations"] = recs
                st.session_state["rec_mode_used"] = "actor"

# ── FAVOURITES MODE ──
elif rec_mode == "favourites":
    st.markdown('<div class="section-header">❤️ Recommendations from Your Favourites</div>', unsafe_allow_html=True)

    if not st.session_state["favourites"]:
        st.info("👈 Add movies to your favourites in the sidebar first")
    else:
        st.markdown(f"**Based on:** {', '.join([f'*{t}*' for t in st.session_state['favourites']])}")
        with st.spinner("🧠 Blending your taste profile…"):
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

# ── COLLABORATIVE FILTERING MODE ──
elif rec_mode == "cf":
    st.markdown('<div class="section-header">👥 Similar User Recommendations</div>', unsafe_allow_html=True)

    if user_movie_matrix.empty:
        st.warning("📊 Ratings data not available — use Search or Favourites mode instead")
    else:
        available_users = sorted(user_movie_matrix.index.tolist())
        selected_user = st.selectbox(
            "Simulate recommendations for user:",
            options=available_users,
            index=0,
        )
        st.session_state["selected_user"] = selected_user

        if st.button("👤 Get User-Based Recommendations", type="primary", use_container_width=True):
            with st.spinner("🔍 Finding similar users…"):
                recs = get_recommendations_for_user(
                    user_id=selected_user,
                    user_movie_matrix=user_movie_matrix,
                    movies_df=movies_df,
                    similarity_matrix=sim_matrix,
                    n=n_recs,
                )
                st.session_state["recommendations"] = recs
                st.session_state["rec_mode_used"] = "cf"

        with st.expander("📊 User's Ratings"):
            user_rated = user_movie_matrix.loc[selected_user].dropna().sort_values(ascending=False)
            st.dataframe(
                user_rated.reset_index().rename(columns={"index": "Movie", selected_user: "Rating"}),
                use_container_width=True,
                hide_index=True,
            )


# ──────────────────────────────────────────────
# Results Display
# ──────────────────────────────────────────────

recs: pd.DataFrame | None = st.session_state.get("recommendations")

if recs is not None and not recs.empty:
    mode_used = st.session_state.get("rec_mode_used", "cbf_single")
    
    # Determine score column and label
    if mode_used == "actor":
        st.markdown(f'<div class="section-header">📽 Found {len(recs)} Movie(s)</div>', unsafe_allow_html=True)
        show_score = False
    else:
        label_map = {
            "cbf_single": "Content Similarity",
            "cbf_multi":  "Blended Taste Score",
            "cf":         "Predicted Rating",
        }
        score_label = label_map.get(mode_used, "Score")
        st.markdown(f'<div class="section-header">✨ Top Results — {score_label}</div>', unsafe_allow_html=True)
        show_score = True

    # 3-column grid
    cols = st.columns(3)

    for i, (_, row) in enumerate(recs.iterrows()):
        col = cols[i % 3]
        with col:
            with st.container():
                # Poster image
                if api_key_configured():
                    meta = get_tmdb_metadata(row["title"], int(row.get("year", 0)))
                    poster_url = meta["poster_url"]
                else:
                    poster_url = None

                if poster_url:
                    st.image(poster_url, use_column_width=True)

                # Title and year
                st.markdown(f"### {row['title']}")
                st.caption(f"📅 {int(row['year'])}")

                # Genre tags
                if "genre" in row and row["genre"]:
                    genre_html = " ".join(
                        f'<span class="genre-tag">{g.strip()}</span>'
                        for g in str(row.get("genre", "")).split()
                        if g.strip()
                    )
                    st.markdown(genre_html, unsafe_allow_html=True)

                # Score badge
                if show_score and "similarity_score" in row:
                    score_val = row["similarity_score"]
                    st.markdown(
                        f'<span class="score-badge">Score: {score_val:.2f}</span>',
                        unsafe_allow_html=True,
                    )
                elif show_score and "predicted_rating" in row:
                    score_val = row["predicted_rating"]
                    st.markdown(
                        f'<span class="score-badge">Predicted: {score_val:.2f}⭐</span>',
                        unsafe_allow_html=True,
                    )

                # Overview
                with st.expander("📖 Plot"):
                    st.caption(row.get("overview", "No description available"))

                # Cast info (if available)
                if "cast" in row and row["cast"]:
                    with st.expander("👥 Cast"):
                        st.caption(row["cast"])

                # Add to favourites button
                if row["title"] not in st.session_state["favourites"]:
                    if st.button(f"💕 Add to Favourites", key=f"add_fav_{i}", use_container_width=True):
                        st.session_state["favourites"].append(row["title"])
                        st.toast(f"Added to favourites! ❤️", icon="✅")
                        st.rerun()

                st.markdown("---")

elif recs is not None and recs.empty:
    st.info("🤔 No recommendations found. Try a different search or add more favourites.")
