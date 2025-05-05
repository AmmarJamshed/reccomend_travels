#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------
# Synthetic Data Generation
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
    "Mindful Seeker": ["Bali, Indonesia", "Kyoto, Japan", "Kerala, India"],
    "Curious Connector": ["Lisbon, Portugal", "Istanbul, Turkey", "Buenos Aires, Argentina"],
    "Independent Explorer": ["New Zealand", "Iceland", "Scotland Highlands"],
    "Earth Lover": ["Costa Rica", "Norwegian Fjords", "Patagonia, Chile"],
    "Elegant Voyager": ["Paris, France", "Vienna, Austria", "Florence, Italy"],
    "Cultural Alchemist": ["Marrakech, Morocco", "Lahore, Pakistan", "Hanoi, Vietnam"],
    "Trailblazing Energizer": ["Peru", "South Africa", "Arizona, USA"],
    "Heartful Healer": ["Sedona, USA", "Rishikesh, India", "Ubud, Bali"],
    "Radiant Nomad": ["Thailand", "Mexico", "Portugal"],
    "Structured Nomad": ["Germany", "Singapore", "Canada"],
    "Wild Mystic": ["Amazon Rainforest", "Tibet", "Madagascar"],
    "Offbeat Nomad": ["Georgia (Tbilisi)", "Uzbekistan", "Bhutan"],
    "Sensory Wanderer": ["Italy", "Morocco", "Thailand"],
    "Inner Voyager": ["Nepal", "Sri Lanka", "Greece"],
    "Time Traveler": ["Rome", "Cairo", "Athens"],
    "Sacred Pilgrim": ["Mecca", "Varanasi", "Jerusalem"],
    "Urban Soulwalker": ["New York City", "Berlin", "Tokyo"]
}

# Generate synthetic user data
def generate_user():
    num_badges = random.randint(3, 7)
    return random.sample(badges, num_badges)

df = pd.DataFrame({
    "badges": [generate_user() for _ in range(200)],
    "archetype": [random.choice(archetypes) for _ in range(200)]
})

# Encode badges
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["badges"])
y = df["archetype"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# --------------------------
# Streamlit App UI
# --------------------------

st.set_page_config(page_title="Erranza Travel Engine", layout="wide")
st.image("https://images.unsplash.com/photo-1507525428034-b723cf961d3e", use_column_width=True)
st.title("🌍 Erranza Travel Archetype Finder")
st.subheader("Find your travel vibe, get personalized archetype & destination recommendations.")

# Sidebar
st.sidebar.header("Model Info")
st.sidebar.metric("Model Accuracy", f"{accuracy*100:.2f}%")
st.sidebar.markdown("Trained on 200 synthetic users using Logistic Regression.")

# User Input
st.markdown("### Select the badges that reflect your travel preferences:")
selected = st.multiselect("Pick your travel badges:", badges)

if st.button("🔮 Get My Travel Archetype & Places"):
    if not selected:
        st.error("Please select at least one badge.")
    else:
        # Predict
        input_vector = mlb.transform([selected])
        predicted_archetype = model.predict(input_vector)[0]
        st.success(f"🎯 Your Travel Archetype: **{predicted_archetype}**")

        st.markdown("### ✈️ Recommended Destinations:")
        recs = recommendations.get(predicted_archetype, [])
        for place in recs:
            st.markdown(f"- {place}")

        st.image("https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0", caption="Let your next journey begin...", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("Built by [Ammar Jamshed](https://linkedin.com/in/ammarjamshed) • Powered by synthetic ML")

