"""
main.py  –  CineMatch (Netflix Edition)
-----------------------------------------
Netflix-inspired UI with:
  - Dark #141414 background
  - Official TMDB posters
  - YouTube trailer embeds (official trailers via TMDB videos endpoint)
  - Horizontal scrollable movie rows by genre
  - Hero section with featured movie
  - Search, Actor, Favourites and CF modes
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

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Netflix CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Root colours ── */
  :root {
    --netflix-black:  #141414;
    --netflix-dark:   #181818;
    --netflix-card:   #2F2F2F;
    --netflix-red:    #E50914;
    --netflix-hover:  #FF0A16;
    --text-primary:   #FFFFFF;
    --text-secondary: #B3B3B3;
  }

  /* ── Full-page dark background ── */
  [data-testid="stAppViewContainer"],
  [data-testid="stHeader"],
  section.main { background: var(--netflix-black) !important; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #000 !important;
    border-right: 1px solid #333;
  }

  /* ── Hide default Streamlit chrome ── */
  footer, #MainMenu { visibility: hidden; }

  /* ── Netflix-style navbar brand ── */
  .nf-brand {
    font-size: 2.2rem;
    font-weight: 900;
    color: var(--netflix-red);
    letter-spacing: -1px;
    font-family: 'Arial Black', sans-serif;
    margin-bottom: 0;
  }
  .nf-tagline {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-top: -8px;
    margin-bottom: 16px;
  }

  /* ── Hero banner ── */
  .hero-banner {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 32px;
    background: linear-gradient(
      to right,
      rgba(20,20,20,0.95) 0%,
      rgba(20,20,20,0.7)  40%,
      rgba(20,20,20,0.1)  100%
    );
  }
  .hero-backdrop {
    width: 100%;
    height: 420px;
    object-fit: cover;
    border-radius: 12px;
    display: block;
  }
  .hero-content {
    position: absolute;
    top: 0; left: 0;
    width: 50%;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 32px;
    background: linear-gradient(
      to right,
      rgba(20,20,20,1) 0%,
      rgba(20,20,20,0.8) 70%,
      transparent 100%
    );
    border-radius: 12px;
  }
  .hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
  }
  .hero-meta {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-bottom: 12px;
  }
  .hero-overview {
    color: #ccc;
    font-size: 0.9rem;
    line-height: 1.5;
    margin-bottom: 20px;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  /* ── Row headings ── */
  .row-heading {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 24px 0 12px 0;
    border-left: 4px solid var(--netflix-red);
    padding-left: 12px;
  }

  /* ── Movie card ── */
  .movie-card-wrap {
    position: relative;
    border-radius: 6px;
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    background: var(--netflix-card);
  }
  .movie-card-wrap:hover {
    transform: scale(1.08);
    box-shadow: 0 16px 40px rgba(0,0,0,0.8);
    z-index: 10;
  }
  .movie-card-img {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
  }
  .movie-card-overlay {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    background: linear-gradient(transparent 0%, rgba(0,0,0,0.95) 100%);
    padding: 32px 10px 10px;
    opacity: 0;
    transition: opacity 0.3s ease;
  }
  .movie-card-wrap:hover .movie-card-overlay { opacity: 1; }
  .card-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: white;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .card-meta {
    font-size: 0.72rem;
    color: var(--text-secondary);
  }
  .card-genre {
    display: inline-block;
    font-size: 0.65rem;
    background: var(--netflix-red);
    color: white;
    padding: 1px 6px;
    border-radius: 3px;
    margin-top: 4px;
  }

  /* ── Score pill ── */
  .score-pill {
    display: inline-block;
    background: var(--netflix-red);
    color: white;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 4px 4px 0 0;
  }
  .score-pill.gold {
    background: #F5C518;
    color: #000;
  }

  /* ── Trailer iframe ── */
  .trailer-wrap {
    border-radius: 10px;
    overflow: hidden;
    border: 2px solid var(--netflix-red);
    margin-top: 12px;
  }
  .trailer-wrap iframe {
    width: 100%;
    height: 280px;
    display: block;
    border: none;
  }

  /* ── Inputs ── */
  .stTextInput input {
    background: #333 !important;
    color: white !important;
    border: 1px solid #555 !important;
    border-radius: 6px !important;
  }
  .stTextInput input:focus {
    border-color: var(--netflix-red) !important;
    box-shadow: 0 0 0 2px rgba(229,9,20,0.3) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: var(--netflix-red) !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 10px 22px !important;
    transition: background 0.2s ease !important;
  }
  .stButton > button:hover {
    background: var(--netflix-hover) !important;
  }

  /* ── Expander ── */
  details summary {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
  }

  /* ── Divider ── */
  hr { border-color: #333 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🎬 Loading movie database…")
def get_data():
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


# ── Session state ─────────────────────────────────────────────────────────────

for key, val in {
    "favourites":      [],
    "last_search":     "",
    "recommendations": None,
    "rec_mode":        "search",
    "active_movie":    None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Helper: render movie card HTML ───────────────────────────────────────────

def movie_card_html(title: str, year: int, genre: str, poster_url: str, score: float = 0.0, score_label: str = "") -> str:
    first_genre = str(genre).split()[0] if genre else ""
    score_html = f'<div class="score-pill">{score_label}: {score:.2f}</div>' if score > 0 and score_label else ""
    return f"""
    <div class="movie-card-wrap">
      <img class="movie-card-img" src="{poster_url}" alt="{title}" loading="lazy"
           onerror="this.src='https://placehold.co/300x450/141414/E50914?text={title[:12]}'"/>
      <div class="movie-card-overlay">
        <div class="card-title">{title}</div>
        <div class="card-meta">{year}</div>
        <span class="card-genre">{first_genre}</span>
        {score_html}
      </div>
    </div>
    """


# ── Helper: render trailer embed ────────────────────────────────────────────

def render_trailer(trailer_url: str):
    if not trailer_url:
        st.caption("No trailer available")
        return
    st.markdown(
        f"""<div class="trailer-wrap">
              <iframe src="{trailer_url}?autoplay=0&rel=0"
                      allow="accelerometer; clipboard-write; encrypted-media; gyroscope"
                      allowfullscreen></iframe>
            </div>""",
        unsafe_allow_html=True,
    )


# ── Helper: render a row of movie cards ─────────────────────────────────────

def render_movie_row(row_title: str, movies: pd.DataFrame, score_col: str = "", score_label: str = "", max_cols: int = 6):
    if movies.empty:
        return

    st.markdown(f'<div class="row-heading">{row_title}</div>', unsafe_allow_html=True)

    # Slice to max_cols
    movies = movies.head(max_cols)
    cols   = st.columns(len(movies))

    for i, (_, row) in enumerate(movies.iterrows()):
        with cols[i]:
            title  = row["title"]
            year   = int(row.get("year", 0))
            genre  = row.get("genre", "")
            score  = float(row.get(score_col, 0.0)) if score_col and score_col in row else 0.0

            # Get poster from TMDB (or fallback)
            if api_key_configured():
                meta       = get_tmdb_metadata(title, year)
                poster_url = meta["poster_url"]
            else:
                poster_url = f"https://placehold.co/300x450/141414/E50914?text={title[:10].replace(' ', '+')}"

            # Render the poster card
            st.markdown(
                movie_card_html(title, year, genre, poster_url, score, score_label),
                unsafe_allow_html=True,
            )

            # Expandable detail panel below the card
            with st.expander("▶ Details"):
                overview = row.get("overview", "No description available.")
                st.caption(overview)

                if score > 0 and score_label:
                    st.markdown(
                        f'<span class="score-pill">{score_label}: {score:.2f}</span>',
                        unsafe_allow_html=True,
                    )

                # Trailer button
                if api_key_configured():
                    if st.button("🎬 Trailer", key=f"trailer_{row_title}_{i}"):
                        meta = get_tmdb_metadata(title, year)
                        render_trailer(meta.get("trailer_url"))

                # Add to favourites
                if title not in st.session_state["favourites"]:
                    if st.button("💕 Favourite", key=f"fav_{row_title}_{i}"):
                        st.session_state["favourites"].append(title)
                        st.toast(f"Added '{title}' ❤️")
                        st.rerun()


# ── NAVBAR ───────────────────────────────────────────────────────────────────

nav_left, nav_right = st.columns([2, 3])

with nav_left:
    st.markdown('<div class="nf-brand">🎬 CINEMATCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="nf-tagline">Your personal movie discovery engine</div>', unsafe_allow_html=True)

with nav_right:
    nav_cols = st.columns(4)
    labels   = ["🔍 Search", "🎭 Actor", "❤️ Favourites", "👥 Users"]
    modes    = ["search", "actor", "favourites", "cf"]
    for i, (label, mode_key) in enumerate(zip(labels, modes)):
        with nav_cols[i]:
            if st.button(label, use_container_width=True, key=f"nav_{mode_key}"):
                st.session_state["rec_mode"] = mode_key
                st.session_state["recommendations"] = None
                st.rerun()

st.divider()


# ── HERO BANNER (featured movie) ─────────────────────────────────────────────

featured = movies_df.sample(1).iloc[0]

if api_key_configured():
    meta        = get_tmdb_metadata(featured["title"], int(featured["year"]))
    backdrop    = meta["backdrop_url"]
    poster      = meta["poster_url"]
    vote_avg    = meta["vote_average"]
    trailer_url = meta["trailer_url"]
else:
    backdrop    = ""
    poster      = ""
    vote_avg    = 0
    trailer_url = None

if backdrop:
    hero_left, hero_right = st.columns([3, 2])
    with hero_left:
        st.markdown(
            f"""
            <div style="padding: 24px 0;">
              <div style="color:#E50914;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">
                ★ FEATURED TODAY
              </div>
              <div style="font-size:2.4rem;font-weight:900;color:white;line-height:1.1;margin-bottom:10px;">
                {featured['title']}
              </div>
              <div style="color:#B3B3B3;font-size:0.9rem;margin-bottom:12px;">
                {int(featured['year'])} &nbsp;•&nbsp; {featured['genre']}
                {'&nbsp;•&nbsp; ⭐ ' + str(round(vote_avg,1)) if vote_avg else ''}
              </div>
              <div style="color:#ccc;font-size:0.88rem;line-height:1.6;margin-bottom:20px;">
                {str(featured['overview'])[:200]}…
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        btn_a, btn_b = st.columns(2)
        with btn_a:
            if st.button("🎯 Get Recommendations", key="hero_rec", use_container_width=True):
                with st.spinner("Finding similar movies…"):
                    try:
                        recs = get_recommendations_by_movie(
                            title=featured["title"],
                            movies_df=movies_df,
                            similarity_matrix=sim_matrix,
                            n=6,
                        )
                        st.session_state["recommendations"] = recs
                        st.session_state["rec_label"]  = "Similarity Score"
                        st.session_state["rec_col"]    = "similarity_score"
                        st.session_state["rec_heading"] = f"Because you like {featured['title']}"
                    except ValueError:
                        pass
        with btn_b:
            if trailer_url and st.button("▶ Watch Trailer", key="hero_trailer", use_container_width=True):
                render_trailer(trailer_url)

    with hero_right:
        if poster:
            st.markdown(
                f'<img src="{poster}" style="width:100%;border-radius:12px;box-shadow:0 20px 60px rgba(0,0,0,0.8);" />',
                unsafe_allow_html=True,
            )

else:
    # No TMDB key - show text-only hero
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#1a0000 0%,#2a0a0a 100%);
                    border-radius:12px;padding:40px;margin-bottom:24px;
                    border:1px solid #E50914;">
          <div style="color:#E50914;font-size:0.75rem;font-weight:700;letter-spacing:2px;margin-bottom:8px;">
            ★ FEATURED TODAY
          </div>
          <div style="font-size:2.4rem;font-weight:900;color:white;margin-bottom:8px;">
            {featured['title']}
          </div>
          <div style="color:#B3B3B3;margin-bottom:12px;">
            {int(featured['year'])} • {featured['genre']}
          </div>
          <div style="color:#ccc;font-size:0.9rem;line-height:1.6;">
            {str(featured['overview'])[:220]}…
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── MAIN CONTENT MODES ───────────────────────────────────────────────────────

rec_mode = st.session_state["rec_mode"]

# ── SEARCH MODE ──
if rec_mode == "search":
    st.markdown('<div class="row-heading">🔍 Search by Movie Title</div>', unsafe_allow_html=True)

    query = st.text_input(
        "Movie title",
        placeholder="e.g. Inception, The Dark Knight, Parasite…",
        value=st.session_state["last_search"],
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        run = st.button("Get Recommendations", type="primary", use_container_width=True)
    with c2:
        if st.button("Clear", use_container_width=True):
            st.session_state["last_search"] = ""
            st.session_state["recommendations"] = None
            st.rerun()

    if run and query:
        st.session_state["last_search"] = query
        with st.spinner("Analysing…"):
            try:
                recs = get_recommendations_by_movie(query, movies_df, sim_matrix, n=6)
                st.session_state["recommendations"] = recs
                st.session_state["rec_label"]   = "Similarity"
                st.session_state["rec_col"]     = "similarity_score"
                st.session_state["rec_heading"] = f"Because you watched: {query}"
            except ValueError as e:
                st.warning(f"⚠️ {e}")
                matches = search_movies(query, movies_df, 5)
                if not matches.empty:
                    st.markdown("**Did you mean:**")
                    for _, m in matches.iterrows():
                        if st.button(f"📽 {m['title']} ({m['year']})", key=f"s_{m['title']}"):
                            st.session_state["last_search"] = m["title"]
                            st.rerun()

# ── ACTOR SEARCH MODE ──
elif rec_mode == "actor":
    st.markdown('<div class="row-heading">🎭 Find Movies by Actor</div>', unsafe_allow_html=True)

    actor = st.text_input(
        "Actor name",
        placeholder="e.g. Leonardo DiCaprio, Meryl Streep…",
        label_visibility="collapsed",
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        run_actor = st.button("Find Movies", type="primary", use_container_width=True)
    with c2:
        sort_by = st.selectbox("Sort", ["Year (Newest)", "Title (A–Z)"], label_visibility="collapsed")

    if run_actor and actor:
        recs = search_by_actor(actor, movies_df)
        if recs.empty:
            st.warning(f"No movies found with '{actor}'")
        else:
            if sort_by == "Title (A–Z)":
                recs = recs.sort_values("title").reset_index(drop=True)
            st.session_state["recommendations"] = recs
            st.session_state["rec_label"]   = ""
            st.session_state["rec_col"]     = ""
            st.session_state["rec_heading"] = f"Movies featuring: {actor}"

# ── FAVOURITES MODE ──
elif rec_mode == "favourites":
    st.markdown('<div class="row-heading">❤️ Based on Your Favourites</div>', unsafe_allow_html=True)

    # Sidebar-style favourites manager in main content
    fav_add = st.selectbox(
        "Add to favourites",
        [""] + [t for t in ALL_TITLES if t not in st.session_state["favourites"]],
    )
    if fav_add:
        st.session_state["favourites"].append(fav_add)
        st.rerun()

    if st.session_state["favourites"]:
        st.markdown(f"**Your list:** {' • '.join(st.session_state['favourites'])}")
        c1, c2 = st.columns([2, 1])
        with c1:
            if st.button("Get Recommendations", type="primary", use_container_width=True):
                with st.spinner("Blending your taste…"):
                    try:
                        recs = get_recommendations_by_titles(
                            st.session_state["favourites"], movies_df, sim_matrix, n=6
                        )
                        st.session_state["recommendations"] = recs
                        st.session_state["rec_label"]   = "Match Score"
                        st.session_state["rec_col"]     = "similarity_score"
                        st.session_state["rec_heading"] = "Top Picks for You"
                    except ValueError as e:
                        st.warning(str(e))
        with c2:
            if st.button("Clear List", use_container_width=True):
                st.session_state["favourites"] = []
                st.rerun()
    else:
        st.info("Add at least one movie above to get recommendations.")

# ── COLLABORATIVE FILTERING ──
elif rec_mode == "cf":
    st.markdown('<div class="row-heading">👥 People Like You Also Watched</div>', unsafe_allow_html=True)

    if user_movie_matrix.empty:
        st.warning("Ratings data unavailable — use Search or Favourites mode.")
    else:
        users = sorted(user_movie_matrix.index.tolist())
        user  = st.selectbox("Select user profile", users)
        if st.button("Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Analysing similar users…"):
                recs = get_recommendations_for_user(
                    user, user_movie_matrix, movies_df, sim_matrix, n=6
                )
                st.session_state["recommendations"] = recs
                st.session_state["rec_label"]   = "Predicted ⭐"
                st.session_state["rec_col"]     = "predicted_rating"
                st.session_state["rec_heading"] = f"Picks for {user}"


# ── RESULTS ROW ──────────────────────────────────────────────────────────────

recs = st.session_state.get("recommendations")
if recs is not None and not recs.empty:
    heading     = st.session_state.get("rec_heading", "Recommendations")
    score_col   = st.session_state.get("rec_col", "")
    score_label = st.session_state.get("rec_label", "")
    render_movie_row(heading, recs, score_col, score_label, max_cols=6)
elif recs is not None and recs.empty:
    st.info("No results found. Try a different title or actor.")


# ── GENRE BROWSE ROWS (Browse All) ───────────────────────────────────────────

st.divider()
st.markdown('<div class="row-heading">🎭 Browse by Genre</div>', unsafe_allow_html=True)

# Extract unique genres
all_genres = set()
for genres_str in movies_df["genre"].dropna():
    for g in str(genres_str).split():
        all_genres.add(g.strip())

for genre in sorted(all_genres):
    genre_movies = movies_df[movies_df["genre"].str.contains(genre, case=False, na=False)]
    if not genre_movies.empty:
        render_movie_row(f"🎬 {genre}", genre_movies, max_cols=6)


# ── FAVOURITES QUICK PANEL ───────────────────────────────────────────────────

if st.session_state["favourites"]:
    st.divider()
    st.markdown('<div class="row-heading">💕 Your Favourites</div>', unsafe_allow_html=True)
    fav_df = movies_df[movies_df["title"].isin(st.session_state["favourites"])]
    render_movie_row("", fav_df, max_cols=6)

    if st.button("🗑 Clear All Favourites"):
        st.session_state["favourites"] = []
        st.rerun()
