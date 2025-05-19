import streamlit as st 
import pandas as pd
import numpy as np
import random
import os
import datetime
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
# User Authentication System
# --------------------------
user_db_file = "users.csv"

def load_users():
    if os.path.exists(user_db_file):
        return pd.read_csv(user_db_file)
    else:
        return pd.DataFrame(columns=["username", "email", "password"])

def save_user(username, email, password):
    users = load_users()
    if username in users["username"].values:
        return False
    new_user = pd.DataFrame([[username, email, password]], columns=["username", "email", "password"])
    new_user.to_csv(user_db_file, mode='a', header=not os.path.exists(user_db_file), index=False)
    return True

def authenticate(username, password):
    users = load_users()
    user_row = users[users["username"] == username]
    if not user_row.empty and user_row.iloc[0]["password"] == password:
        return True
    return False

# --------------------------
# Generate Training Data
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
# Save to CSV
# --------------------------
def append_to_csv(data, file_path="soulprint_log.csv"):
    columns = ["user_name", "timestamp", "selected_badges", "assigned_archetype"]
    df_new = pd.DataFrame([data], columns=columns)

    if os.path.exists(file_path):
        df_new.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(file_path, mode='w', header=True, index=False)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Erranza AI Travel Companion", layout="wide")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2555/2555028.png", width=100)
st.sidebar.title("✈️ Explore the World with Erranza AI")

# Authentication
st.sidebar.markdown("## 👤 Account Access")
auth_mode = st.sidebar.radio("Select Mode", ["Login", "Register"])
username = st.sidebar.text_input("Username")
email = st.sidebar.text_input("Email") if auth_mode == "Register" else ""
password = st.sidebar.text_input("Password", type="password")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = ""

if auth_mode == "Login":
    if st.sidebar.button("🔓 Login"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success(f"Welcome back, {username}!")
        else:
            st.error("Invalid credentials. Try again.")
elif auth_mode == "Register":
    if st.sidebar.button("📝 Register"):
        if save_user(username, email, password):
            st.success("Account created! You can now log in.")
        else:
            st.warning("Username already exists.")

# Logged-in user view
if st.session_state.logged_in:
    name = st.session_state.user
    selected = st.sidebar.multiselect("Select Your Travel Personality Badges:", badges)
    get_recs = st.sidebar.button("✨ Get Recommendations")
else:
    st.sidebar.info("Please login or register to use the recommendation system.")
    selected = []
    get_recs = False

# ML Model
df = get_data()
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["badges"])
y = df["archetype"]
model = RandomForestClassifier()
model.fit(X, y)

if get_recs and name:
    if not selected:
        st.warning("Please select at least one badge.")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/201/201623.png", width=120, caption="🌐 Your AI Travel Guide")
        st.subheader(f"🧭 Welcome, {name}!")
        input_vector = mlb.transform([selected])
        pred = model.predict(input_vector)[0]
        st.success(f"🎯 Your Travel Archetype: **{pred}**")

        # Save session to CSV
        timestamp = datetime.datetime.utcnow().isoformat()
        session_row = {
            "user_name": name,
            "timestamp": timestamp,
            "selected_badges": ", ".join(selected),
            "assigned_archetype": pred
        }
        append_to_csv(session_row)

        # Recommendations
        st.markdown("### ✈️ Suggested Destinations:")
        recs = recommendations.get(pred, [])
        for country, city in recs:
            st.markdown(f"- 🌍 **{country}**, ✈️ _{city}_")

        # Similarity Matching
        st.markdown("### 🔍 Most Similar Users:")
        sim_scores = cosine_similarity(input_vector, X).flatten()
        top_indices = np.argsort(sim_scores)[-3:][::-1]
        for i in top_indices:
            badges_similar = ", ".join(df.iloc[i]["badges"])
            archetype_similar = df.iloc[i]["archetype"]
            st.markdown(f"**User {i+1}**: {archetype_similar} | _{badges_similar}_")

        st.markdown("---")
        st.caption("Built for Erranza.ai")
