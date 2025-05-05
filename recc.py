import streamlit as st
import pandas as pd
import numpy as np
import random
import mysql.connector
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# --------------------------
# Static Data
# --------------------------
badges = [
    "Inner + Flow", "Nature + Stillness", "Culture + Connection", "Community & Local First",
    "Food + Soul", "Wonder + Mystery", "Solo Traveler", "Off the Path", "Refinement + Aesthetics",
    "Luxury/Refined", "Budget-Friendly", "Adrenaline + Wild", "Urban + Discovery", "Slow & Soulful",
    "Heritage + History", "Holistic Ethical Travel", "Craft + Creation", "Journal to Self"
]

archetypes = [
    "Mindful Seeker", "Curious Connector", "Independent Explorer", "Earth Lover", "Elegant Voyager",
    "Cultural Alchemist", "Trailblazing Energizer", "Heartful Healer", "Radiant Nomad",
    "Structured Nomad", "Wild Mystic", "Offbeat Nomad", "Sensory Wanderer", "Inner Voyager",
    "Time Traveler", "Sacred Pilgrim", "Urban Soulwalker"
]

recommendations = {
    "Mindful Seeker": ["Bali", "Kyoto", "Kerala"],
    "Curious Connector": ["Lisbon", "Istanbul", "Buenos Aires"],
    "Independent Explorer": ["New Zealand", "Iceland", "Scotland"],
    "Earth Lover": ["Costa Rica", "Norwegian Fjords", "Patagonia"],
    "Elegant Voyager": ["Paris", "Vienna", "Florence"],
    "Cultural Alchemist": ["Marrakech", "Lahore", "Hanoi"],
    "Trailblazing Energizer": ["Peru", "South Africa", "Arizona"],
    "Heartful Healer": ["Sedona", "Rishikesh", "Ubud"],
    "Radiant Nomad": ["Thailand", "Mexico", "Portugal"],
    "Structured Nomad": ["Germany", "Singapore", "Canada"],
    "Wild Mystic": ["Amazon", "Tibet", "Madagascar"],
    "Offbeat Nomad": ["Tbilisi", "Uzbekistan", "Bhutan"],
    "Sensory Wanderer": ["Italy", "Morocco", "Thailand"],
    "Inner Voyager": ["Nepal", "Sri Lanka", "Greece"],
    "Time Traveler": ["Rome", "Cairo", "Athens"],
    "Sacred Pilgrim": ["Mecca", "Varanasi", "Jerusalem"],
    "Urban Soulwalker": ["New York", "Berlin", "Tokyo"]
}

# --------------------------
# Data Handling
# --------------------------
def generate_user():
    return random.sample(badges, random.randint(3, 7))

def get_data():
    # Normally you'd use SQL here
    df = pd.DataFrame({
        "badges": [generate_user() for _ in range(200)],
        "archetype": [random.choice(archetypes) for _ in range(200)]
    })
    return df

def train_models(X, y):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3)
    }
    for name, model in models.items():
        model.fit(X, y)
    return models

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Travel Archetype Engine", layout="wide")
st.title("🌍 Erranza Travel Archetype Finder")
st.image("https://images.unsplash.com/photo-1507525428034-b723cf961d3e", use_column_width=True)

# Load or generate data
df = get_data()
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["badges"])
y = df["archetype"]

# Train Models
models = train_models(X, y)

# Sidebar
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
st.sidebar.markdown("Trained on 200 synthetic users.")
st.sidebar.markdown("Using cosine similarity to suggest similar users.")

# User Input
selected = st.multiselect("Select badges that describe you:", badges)

if st.button("🔮 Get Recommendations"):
    if not selected:
        st.warning("Please select at least one badge.")
    else:
        input_vector = mlb.transform([selected])
        selected_model = models[model_choice]
        pred = selected_model.predict(input_vector)[0]
        st.success(f"🎯 Your Travel Archetype: **{pred}**")

        # Recommendations
        st.markdown("### ✈️ Suggested Destinations:")
        recs = recommendations.get(pred, [])
        for r in recs:
            st.markdown(f"- {r}")

        # Similarity-based matches
        st.markdown("### 🔍 Most Similar Users:")
        sim_scores = cosine_similarity(input_vector, X).flatten()
        top_indices = np.argsort(sim_scores)[-3:][::-1]

        for i in top_indices:
            badges_similar = ", ".join(df.iloc[i]["badges"])
            archetype_similar = df.iloc[i]["archetype"]
            st.markdown(f"**User {i+1}:** {archetype_similar} | Badges: _{badges_similar}_")

# Footer
st.markdown("---")
st.markdown("Built by [Ammar Jamshed](https://linkedin.com/in/ammarjamshed)")
