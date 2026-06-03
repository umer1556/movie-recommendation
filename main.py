"""
main.py — CineMatch (Netflix Edition)
Tabs-based navigation: active tab is highlighted, content appears in-place.
"""

import streamlit as st
import pandas as pd

from data_loader import (
    load_movies, load_ratings,
    build_similarity_matrix, build_user_movie_matrix,
)
from recommender import (
    get_recommendations_by_movie,
    get_recommendations_by_titles,
    get_recommendations_for_user,
    search_by_actor,
    search_movies,
)
from tmdb_client import get_tmdb_metadata, api_key_configured

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Base ── */
  :root {
    --nf-black:  #141414;
    --nf-dark:   #181818;
    --nf-card:   #2F2F2F;
    --nf-red:    #E50914;
    --nf-white:  #FFFFFF;
    --nf-grey:   #B3B3B3;
  }

  [data-testid="stAppViewContainer"],
  [data-testid="stHeader"],
  section.main { background: var(--nf-black) !important; }

  footer, #MainMenu { visibility: hidden; }

  /* ── Brand ── */
  .nf-brand {
    font-size: 2.2rem;
    font-weight: 900;
    color: var(--nf-red);
    font-family: 'Arial Black', sans-serif;
    letter-spacing: -1px;
  }

  /* ── Tab bar styling ──────────────────────────────────────────
     Make the tabs look like a Netflix-style top nav.
     Active tab = red underline + white text.
     Inactive tab = grey text, no underline.
  ── */
  [data-testid="stTabs"] > div:first-child {
    gap: 0 !important;
    border-bottom: 1px solid #333 !important;
    margin-bottom: 20px;
  }

  /* Each tab button */
  [data-testid="stTabs"] button[role="tab"] {
    background: transparent !important;
    color: var(--nf-grey) !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    transition: color 0.2s, border-color 0.2s;
  }

  /* Hover state */
  [data-testid="stTabs"] button[role="tab"]:hover {
    color: var(--nf-white) !important;
    border-bottom: 3px solid #555 !important;
  }

  /* Active tab — red underline, white text */
  [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--nf-white) !important;
    border-bottom: 3px solid var(--nf-red) !important;
    background: transparent !important;
  }

  /* ── Row headings ── */
  .row-heading {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--nf-white);
    margin: 12px 0 16px 0;
    border-left: 4px solid var(--nf-red);
    padding-left: 12px;
  }

  /* ── Movie cards ── */
.movie-card-wrap {
    border-radius: 6px;
    overflow: hidden;
    background: linear-gradient(145deg, #2a2a2a, #1a1a1a);
    max-width: 100%;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
  .movie-card-wrap:hover {
    transform: scale(1.07);
    box-shadow: 0 12px 36px rgba(0,0,0,0.8);
  }
.movie-card-img {
    width: 100%;
    height: 240px;
    max-height: 240px;
    object-fit: cover;
    object-position: center top;
    display: block;
}
  .movie-card-overlay {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,0.95));
    padding: 32px 10px 10px;
    opacity: 0;
    transition: opacity 0.25s;
  }
  .movie-card-wrap:hover .movie-card-overlay { opacity: 1; }
  .card-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: white;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .card-meta { font-size: 0.72rem; color: var(--nf-grey); }
  .card-genre {
    display: inline-block;
    font-size: 0.65rem;
    background: var(--nf-red);
    color: white;
    padding: 1px 6px;
    border-radius: 3px;
    margin-top: 4px;
  }

  /* ── Score pill ── */
  .score-pill {
    display: inline-block;
    background: var(--nf-red);
    color: white;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 4px 4px 0 0;
  }

  /* ── Trailer ── */
  .trailer-wrap {
    border-radius: 10px;
    overflow: hidden;
    border: 2px solid var(--nf-red);
    margin-top: 10px;
  }
  .trailer-wrap iframe {
    width: 100%;
    height: 260px;
    display: block;
    border: none;
  }

  /* ── Inputs ── */
  .stTextInput input {
    background: #222 !important;
    color: white !important;
    border: 1px solid #444 !important;
    border-radius: 6px !important;
  }
  .stTextInput input:focus {
    border-color: var(--nf-red) !important;
    box-shadow: 0 0 0 2px rgba(229,9,20,0.25) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: var(--nf-red) !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 700 !important;
  }
  .stButton > button:hover {
    background: #FF0A16 !important;
  }

  hr { border-color: #333 !important; }

  /* Disable Streamlit's click-to-fullscreen on images */
[data-testid="stImage"] img {
    pointer-events: none !important;
    cursor: default !important;
    border-radius: 6px !important;
    width: 100% !important;
}

/* Hide the expand button that appears on hover */
[data-testid="stImage"] button,
[data-testid="StyledFullScreenButton"] {
    display: none !important;
}

/* Stop Streamlit markdown container from adding padding around cards */
[data-testid="stMarkdownContainer"] > div {
    line-height: 0;
    font-size: 0;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🎬 Loading CineMatch…")
def get_data():
    movies_df         = load_movies()
    ratings_df        = load_ratings()
    similarity_matrix = build_similarity_matrix(movies_df)
    user_movie_matrix = build_user_movie_matrix(movies_df, ratings_df)
    return movies_df, ratings_df, similarity_matrix, user_movie_matrix


try:
    movies_df, ratings_df, sim_matrix, user_movie_matrix = get_data()
except FileNotFoundError as e:
    st.error(f"Data error: {e}")
    st.stop()

ALL_TITLES = movies_df["title"].sort_values().tolist()


# ── Session state ─────────────────────────────────────────────────────────────

for key, val in {
    "favourites":      [],
    "recommendations": None,
    "rec_label":       "",
    "rec_col":         "",
    "rec_heading":     "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_trailer(trailer_url: str):
    if not trailer_url:
        st.caption("No trailer available for this title.")
        return
    st.markdown(
        f"""<div class="trailer-wrap">
              <iframe src="{trailer_url}?rel=0"
                      allow="accelerometer; clipboard-write; encrypted-media; gyroscope"
                      allowfullscreen></iframe>
            </div>""",
        unsafe_allow_html=True,
    )


def movie_card_html(title, year, genre, poster_url, score=0.0, score_label=""):
    first_genre = str(genre).split()[0] if genre else ""
    score_html  = (
        f'<div class="score-pill">{score_label}: {score:.2f}</div>'
        if score > 0 and score_label else ""
    )
    safe_title = title[:12].replace("'", "")
    return f"""
    <div class="movie-card-wrap" style="position:relative;">
      <img class="movie-card-img"
           src="{poster_url}"
           alt="{title}"
           loading="lazy"
           onerror="this.src='https://placehold.co/300x450/141414/E50914?text={safe_title}'"/>
      <div class="movie-card-overlay">
        <div class="card-title">{title}</div>
        <div class="card-meta">{year}</div>
        <span class="card-genre">{first_genre}</span>
        {score_html}
      </div>
    </div>"""


def render_movie_row(heading, movies_df_row, score_col="", score_label="", max_cols=6):
    if movies_df_row.empty:
        return
    if heading:
        st.markdown(f'<div class="row-heading">{heading}</div>', unsafe_allow_html=True)

    movies_df_row = movies_df_row.head(max_cols)
    cols = st.columns(len(movies_df_row))

    for i, (_, row) in enumerate(movies_df_row.iterrows()):
        with cols[i]:
            title = row["title"]
            year  = int(row.get("year", 0))
            genre = row.get("genre", "")
            score = float(row.get(score_col, 0.0)) if score_col and score_col in row else 0.0

            if api_key_configured():
                meta       = get_tmdb_metadata(title, year)
                poster_url = meta["poster_url"]
            else:
                poster_url = (
                    f"https://placehold.co/300x450/141414/E50914?"
                    f"text={title[:10].replace(' ', '+')}"
                )

            st.markdown(
                movie_card_html(title, year, genre, poster_url, score, score_label),
                unsafe_allow_html=True,
            )

            with st.expander("▶ Details"):
                st.caption(row.get("overview", "No description available."))
                if score > 0 and score_label:
                    st.markdown(
                        f'<span class="score-pill">{score_label}: {score:.2f}</span>',
                        unsafe_allow_html=True,
                    )
                if api_key_configured():
                    if st.button("🎬 Watch Trailer", key=f"t_{heading}_{i}"):
                        meta = get_tmdb_metadata(title, year)
                        render_trailer(meta.get("trailer_url"))
                if row["title"] not in st.session_state["favourites"]:
                    if st.button("💕 Add to Favourites", key=f"f_{heading}_{i}"):
                        st.session_state["favourites"].append(row["title"])
                        st.toast(f"Added '{title}' ❤️")
                        st.rerun()


# ── Brand header ──────────────────────────────────────────────────────────────

st.markdown('<div class="nf-brand">🎬 CINEMATCH</div>', unsafe_allow_html=True)
st.caption("Your personal movie discovery engine")
st.divider()


# ── TABS NAV (replaces dead buttons) ─────────────────────────────────────────
# st.tabs shows the active tab with a red underline automatically.
# Content renders immediately below — no scrolling needed.

tab_search, tab_actor, tab_favs, tab_cf = st.tabs([
    "🔍  Search by Title",
    "🎭  Search by Actor",
    "❤️  My Favourites",
    "👥  Similar Users",
])


# ── TAB 1: SEARCH BY TITLE ────────────────────────────────────────────────────

with tab_search:
    st.markdown('<div class="row-heading">Find movies similar to one you love</div>', unsafe_allow_html=True)

    query = st.text_input(
        "Movie title",
        placeholder="e.g. Inception, The Dark Knight, Parasite…",
        key="search_query",
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        run_search = st.button("Get Recommendations", type="primary", use_container_width=True, key="btn_search")
    with c2:
        if st.button("Clear", use_container_width=True, key="btn_clear_search"):
            st.session_state["recommendations"] = None
            st.rerun()

    if run_search and query:
        with st.spinner("Analysing…"):
            try:
                recs = get_recommendations_by_movie(query, movies_df, sim_matrix, n=6)
                st.session_state.update({
                    "recommendations": recs,
                    "rec_col":         "similarity_score",
                    "rec_label":       "Similarity",
                    "rec_heading":     f"Because you watched: {query}",
                })
            except ValueError as e:
                st.warning(f"⚠️ {e}")
                matches = search_movies(query, movies_df, 5)
                if not matches.empty:
                    st.markdown("**Did you mean:**")
                    for _, m in matches.iterrows():
                        if st.button(f"📽 {m['title']} ({m['year']})", key=f"sug_{m['title']}"):
                            st.session_state["search_query"] = m["title"]
                            st.rerun()

    # Results render in-place, inside this tab
    recs = st.session_state.get("recommendations")
    if recs is not None and not recs.empty:
        render_movie_row(
            st.session_state["rec_heading"],
            recs,
            st.session_state["rec_col"],
            st.session_state["rec_label"],
        )
    elif recs is not None and recs.empty:
        st.info("No results found. Try a different title.")


# ── TAB 2: SEARCH BY ACTOR ───────────────────────────────────────────────────

with tab_actor:
    st.markdown('<div class="row-heading">Find all movies featuring an actor</div>', unsafe_allow_html=True)

    actor_query = st.text_input(
        "Actor name",
        placeholder="e.g. Leonardo DiCaprio, Meryl Streep, Tom Hanks…",
        key="actor_query",
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        run_actor = st.button("Find Movies", type="primary", use_container_width=True, key="btn_actor")
    with c2:
        sort_by = st.radio("Sort by", ["Year (Newest)", "Title (A–Z)"],
                           horizontal=True, key="actor_sort", label_visibility="collapsed")

    if run_actor and actor_query:
        with st.spinner(f"Searching for {actor_query}…"):
            results = search_by_actor(actor_query, movies_df)

        if results.empty:
            st.warning(f"No movies found featuring '{actor_query}'. Check the spelling.")
        else:
            if sort_by == "Title (A–Z)":
                results = results.sort_values("title").reset_index(drop=True)

            st.success(f"Found **{len(results)}** movie(s) featuring *{actor_query}*")
            render_movie_row("", results, max_cols=6)


# ── TAB 3: FAVOURITES ────────────────────────────────────────────────────────

with tab_favs:
    st.markdown('<div class="row-heading">Get recommendations from movies you love</div>', unsafe_allow_html=True)

    # Add to favourites
    new_fav = st.selectbox(
        "Add a movie to your list",
        [""] + [t for t in ALL_TITLES if t not in st.session_state["favourites"]],
        key="fav_picker",
    )
    if new_fav:
        st.session_state["favourites"].append(new_fav)
        st.rerun()

    if st.session_state["favourites"]:
        # Show current list
        st.markdown(f"**Your list ({len(st.session_state['favourites'])} movies):**")
        for i, fav in enumerate(st.session_state["favourites"]):
            col_name, col_rm = st.columns([5, 1])
            col_name.markdown(f"🎥 {fav}")
            if col_rm.button("✕", key=f"rm_{i}"):
                st.session_state["favourites"].pop(i)
                st.rerun()

        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("Get Recommendations", type="primary", use_container_width=True, key="btn_favs"):
                with st.spinner("Blending your taste profile…"):
                    try:
                        recs = get_recommendations_by_titles(
                            st.session_state["favourites"], movies_df, sim_matrix, n=6
                        )
                        st.session_state.update({
                            "recommendations": recs,
                            "rec_col":         "similarity_score",
                            "rec_label":       "Match",
                            "rec_heading":     "Top Picks for You",
                        })
                    except ValueError as e:
                        st.warning(str(e))
        with c2:
            if st.button("Clear all", use_container_width=True, key="btn_clear_favs"):
                st.session_state["favourites"] = []
                st.rerun()

        recs = st.session_state.get("recommendations")
        if recs is not None and not recs.empty and st.session_state.get("rec_heading") == "Top Picks for You":
            render_movie_row("Top Picks for You", recs, "similarity_score", "Match")
    else:
        st.info("👆 Add at least one movie above to get recommendations.")


# ── TAB 4: SIMILAR USERS (CF) ────────────────────────────────────────────────

with tab_cf:
    st.markdown('<div class="row-heading">What people like you also watched</div>', unsafe_allow_html=True)

    if user_movie_matrix.empty:
        st.warning("Ratings data not available. Use Search or Favourites instead.")
    else:
        users = sorted(user_movie_matrix.index.tolist())
        user  = st.selectbox("Select a user profile to simulate:", users, key="cf_user")

        if st.button("Get Recommendations", type="primary", use_container_width=True, key="btn_cf"):
            with st.spinner("Analysing similar users…"):
                recs = get_recommendations_for_user(
                    user, user_movie_matrix, movies_df, sim_matrix, n=6
                )
            render_movie_row(f"Picks for {user}", recs, "predicted_rating", "Predicted ⭐")

        with st.expander(f"📊 {user}'s rated movies"):
            user_rated = (
                user_movie_matrix.loc[user]
                .dropna()
                .sort_values(ascending=False)
                .reset_index()
            )
            user_rated.columns = ["Movie", "Rating"]
            st.dataframe(user_rated, use_container_width=True, hide_index=True)


# ── BROWSE BY GENRE ───────────────────────────────────────────────────────────

st.divider()
st.markdown('<div class="row-heading">🎭 Browse by Genre</div>', unsafe_allow_html=True)

all_genres = sorted({
    g.strip()
    for genres_str in movies_df["genre"].dropna()
    for g in str(genres_str).split()
    if g.strip()
})

for genre in all_genres:
    genre_movies = movies_df[
        movies_df["genre"].str.contains(genre, case=False, na=False)
    ]
    if not genre_movies.empty:
        render_movie_row(f"🎬 {genre}", genre_movies, max_cols=6)


# ── FAVOURITES QUICK ROW ─────────────────────────────────────────────────────

if st.session_state["favourites"]:
    st.divider()
    fav_df = movies_df[movies_df["title"].isin(st.session_state["favourites"])]
    render_movie_row("💕 Your Favourites", fav_df, max_cols=6)
