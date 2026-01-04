import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# Set page to wide mode for a better "Analytical" feel
st.set_page_config(page_title="Zomato Global Analytics", layout="wide")

# 1. Load Model and Scaler (Make sure these files are uploaded to HF)
@st.cache_resource
def load_assets():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    # Optional: Load a small sample of your CSV for the charts
    df = pd.read_csv('cleaned_sample.csv') 
    return model, scaler, df

try:
    model, scaler, df = load_assets()
except:
    st.error("Please upload 'best_model.pkl', 'scaler.pkl', and 'cleaned_sample.csv' to your Space.")
    st.stop()

# Dashboard Title
st.title("📊 Global Restaurant Market: Prediction & Analytics")
st.markdown("---")

# Create Tabs
tab1, tab2 = st.tabs(["🎯 Rating Predictor", "📈 Market Insights"])

# --- TAB 1: PREDICTION DASHBOARD ---
with tab1:
    st.header("Predict Restaurant Success")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Configure Restaurant Profile")
        cost = st.slider("Average Cost for Two (USD)", 1, 500, 25)
        votes = st.number_input("Expected Popularity (Votes)", 0, 10000, 150)
        price_range = st.selectbox("Price Tier (1: Budget, 4: Luxury)", [1, 2, 3, 4])
        cuisines = st.slider("Cuisine Variety", 1, 10, 3)
        online = st.radio("Online Delivery", ["Yes", "No"])
        booking = st.radio("Table Booking", ["Yes", "No"])

    with col2:
        st.subheader("Prediction Result")
        # Process input
        input_data = pd.DataFrame({
            'Average_Cost_USD': [cost],
            'Price range': [price_range],
            'Votes': [votes],
            'Has Table booking': [1 if booking == "Yes" else 0],
            'Has Online delivery': [1 if online == "Yes" else 0],
            'Cuisine_Count': [cuisines]
        })
        
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        
        # Display Gauge or Big Number
        st.metric(label="Predicted Aggregate Rating", value=f"{prediction:.2f} / 5.0")
        
        if prediction >= 4.0:
            st.success("High Success Probability: This configuration matches top-tier global standards.")
        elif prediction >= 3.0:
            st.warning("Average Performance: Competitive but requires better engagement.")
        else:
            st.error("Low Performance: Consider adding delivery services or improving value-for-money.")

# --- TAB 2: ANALYTICAL DASHBOARD ---
with tab2:
    st.header("Global Market Trends")
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Cost vs. Rating")
        fig1 = px.scatter(df, x="Average_Cost_USD", y="Aggregate rating", 
                         color="Price range", trendline="ols",
                         title="Does Higher Price mean Higher Rating?")
        st.plotly_chart(fig1, use_container_width=True)

    with row1_col2:
        st.subheader("Popularity Impact")
        fig2 = px.density_heatmap(df, x="Votes", y="Aggregate rating", 
                                 title="Density of Ratings by Vote Count",
                                 nbinsx=30, nbinsy=30, color_continuous_scale='Viridis')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Service Impact Analysis")
    fig3 = px.box(df, x="Has Online delivery", y="Aggregate rating", color="Has Table booking",
                 title="Impact of Digital Services on Customer Satisfaction")
    st.plotly_chart(fig3, use_container_width=True)