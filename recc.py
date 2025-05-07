import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

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
    "Mindful Seeker": [("India", "Rishikesh"), ("Japan", "Kyoto"), ("Indonesia", "Bali")],
    "Curious Connector": [("Portugal", "Lisbon"), ("Turkey", "Istanbul"), ("Argentina", "Buenos Aires")],
    "Independent Explorer": [("New Zealand", "Queenstown"), ("Iceland", "Reykjavik"), ("Scotland", "Edinburgh")],
    "Earth Lover": [("Costa Rica", "Monteverde"), ("Norway", "Fjords"), ("Chile", "Patagonia")],
    "Elegant Voyager": [("France", "Paris"), ("Austria", "Vienna"), ("Italy", "Florence")],
    "Cultural Alchemist": [("Morocco", "Marrakech"), ("Pakistan", "Lahore"), ("Vietnam", "Hanoi")],
    "Trailblazing Energizer": [("Peru", "Cusco"), ("South Africa", "Cape Town"), ("USA", "Arizona")],
    "Heartful Healer": [("USA", "Sedona"), ("India", "Rishikesh"), ("Indonesia", "Ubud")],
    "Radiant Nomad": [("Thailand", "Chiang Mai"), ("Mexico", "Tulum"), ("Portugal", "Porto")],
    "Structured Nomad": [("Germany", "Berlin"), ("Singapore", "Singapore"), ("Canada", "Toronto")],
    "Wild Mystic": [("Brazil", "Amazon"), ("China", "Tibet"), ("Madagascar", "Antananarivo")],
    "Offbeat Nomad": [("Georgia", "Tbilisi"), ("Uzbekistan", "Samarkand"), ("Bhutan", "Thimphu")],
    "Sensory Wanderer": [("Italy", "Rome"), ("Morocco", "Fez"), ("Thailand", "Bangkok")],
    "Inner Voyager": [("Nepal", "Kathmandu"), ("Sri Lanka", "Kandy"), ("Greece", "Santorini")],
    "Time Traveler": [("Italy", "Rome"), ("Egypt", "Cairo"), ("Greece", "Athens")],
    "Sacred Pilgrim": [("Saudi Arabia", "Mecca"), ("India", "Varanasi"), ("Israel", "Jerusalem")],
    "Urban Soulwalker": [("USA", "New York"), ("Germany", "Berlin"), ("Japan", "Tokyo")]
}

# --------------------------
# Data Handling
# --------------------------
def generate_user():
    return random.sample(badges, random.randint(3, 7))

def get_data():
    df = pd.DataFrame({
        "badges": [generate_user() for _ in range(200)],
        "archetype": [random.choice(archetypes) for _ in range(200)]
    })
    return df

# --------------------------
# UI + Logic
# --------------------------
st.set_page_config(page_title="Erranza AI Travel Companion", layout="wide")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2555/2555028.png", width=100)
st.sidebar.title("✈️ Explore the World with Erranza AI")

name = st.sidebar.text_input("Your Name")
selected = st.sidebar.multiselect("Select Your Travel Personality Badges:", badges)
get_recs = st.sidebar.button("✨ Get Recommendations")

df = get_data()
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["badges"])
y = df["archetype"]

model = RandomForestClassifier()
model.fit(X, y)

if get_recs:
    if not selected or not name:
        st.warning("Please enter your name and select at least one badge.")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=120, caption="🌐 Your AI Travel Guide")
        st.subheader(f"🧭 Welcome, {name}!")
        input_vector = mlb.transform([selected])
        pred = model.predict(input_vector)[0]
        st.success(f"🎯 Your Travel Archetype: **{pred}**")

        # Recommendations
        st.markdown("### ✈️ Suggested Destinations:")
        recs = recommendations.get(pred, [])
        for country, city in recs:
            st.markdown(f"- 🌍 **{country}**, ✈️ _{city}_")

        # Similarity-based matches
        st.markdown("### 🔍 Most Similar Users:")
        sim_scores = cosine_similarity(input_vector, X).flatten()
        top_indices = np.argsort(sim_scores)[-3:][::-1]

        for i in top_indices:
            badges_similar = ", ".join(df.iloc[i]["badges"])
            archetype_similar = df.iloc[i]["archetype"]
            st.markdown(f"**User {i+1}**: {archetype_similar} | _{badges_similar}_")

        st.markdown("---")
        st.caption("Built For Erranza.ai")
