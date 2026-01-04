import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Set up the page
st.set_page_config(page_title="Zomato Analytics", layout="wide")

# Load the saved assets
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
data = pd.read_csv('cleaned_sample.csv')

st.title("🍴 Global Restaurant: Prediction & Analytics Dashboard")

# Create Two Sections
col_predict, col_analyze = st.columns([1, 2])

with col_predict:
    st.header("🎯 Prediction Tool")
    # User inputs
    cost = st.number_input("Cost (USD)", value=25)
    votes = st.number_input("Votes", value=100)
    delivery = st.selectbox("Online Delivery", [1, 0])
    booking = st.selectbox("Table Booking", [1, 0])
    cuisines = st.slider("Cuisine Count", 1, 10, 3)
    price_range = st.slider("Price Range", 1, 4, 2)

    if st.button("Predict Rating"):
        # Format the input exactly like training
        features = [[cost, price_range, votes, booking, delivery, cuisines]]
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        st.metric("Predicted Rating", f"{prediction:.2f} / 5.0")

with col_analyze:
    st.header("📈 Market Analytics")
    # Analytical Chart 1: Cost vs Rating
    fig1 = px.scatter(data, x="Average_Cost_USD", y="Aggregate rating", 
                     color="Price range", title="Relationship: Cost vs Rating")
    st.plotly_chart(fig1)

    # Analytical Chart 2: Importance of Votes
    fig2 = px.histogram(data, x="Votes", y="Aggregate rating", 
                       histfunc="avg", title="How Popularity (Votes) Impacts Ratings")
    st.plotly_chart(fig2)