import streamlit as st
import sys
from pathlib import Path

# --- 1. SETUP PATHS ---
# We need to see the 'src' folder, which is the current folder's parent
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from src.inference import MovieRecommender

# --- 2. CONFIGURATION ---
st.set_page_config(page_title="MovieLens AI", layout="wide")
st.title("AI Movie Recommender")

# --- 3. LOAD BRAIN ---
@st.cache_resource
def load_engine():
    model_path = project_root / "models" / "recommender_artifacts.pkl"
    if not model_path.exists():
        return None
    return MovieRecommender(model_path)

engine = load_engine()

if not engine:
    st.error("Model missing. Run 'python train.py' first!")
    st.stop()

# --- 4. UI LOGIC ---
st.sidebar.header("Controls")
mode = st.sidebar.radio("Choose Mode:", ["Personalized Recommendations", "Find Similar Movies"])

if mode == "Personalized Recommendations":
    user_id = st.sidebar.number_input("User ID", min_value=1, value=1)
    
    if st.sidebar.button("Get Recommendations"):
        recs = engine.get_hybrid_recommendations(user_id)
        if recs is not None:
            st.success(f"Top Picks for User {user_id}")
            st.table(recs)
        else:
            st.warning("User not found in database.")

elif mode == "Find Similar Movies":
    movie_title = st.sidebar.text_input("Enter Movie Title", "Toy Story")
    
    if st.sidebar.button("Search"):
        recs = engine.get_similar_movies(movie_title)
        if recs is not None:
            st.info(f"Movies similar to '{movie_title}'")
            st.table(recs)
        else:
            st.error("Movie not found.")