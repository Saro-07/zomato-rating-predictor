import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Setup
st.set_page_config(page_title="Zomato Dual-Intelligence", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv('cleaned_sample.csv')
    return model, scaler, df

model, scaler, df = load_assets()

# Sidebar Mode Switcher
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Go to:", ["Business: Predict & Improve", "Customer: Discover Restaurants"])

# --- MODE 1: BUSINESS OWNER ---
if mode == "Business: Predict & Improve":
    st.title("📊 Business Strategic Dashboard")
    st.write("Use this to simulate how service changes affect your rating.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        cost = st.number_input("Average Cost for Two (USD)", 5, 500, 25)
        votes = st.number_input("Current Votes", 0, 10000, 100)
        delivery = st.selectbox("Online Delivery", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        booking = st.selectbox("Table Booking", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        cuisines = st.slider("Cuisine Variety", 1, 10, 2)
        price_range = st.slider("Price Tier", 1, 4, 2)

    with col2:
        # Business Prediction Logic
        feat = pd.DataFrame([[cost, price_range, votes, booking, delivery, cuisines]], 
                            columns=['Average_Cost_USD', 'Price range', 'Votes', 'Has Table booking', 'Has Online delivery', 'Cuisine_Count'])
        pred = model.predict(scaler.transform(feat))[0]
        st.metric("Predicted Rating", f"{pred:.2f} / 5.0")
        
        # Working Function: Advice
        if pred < 3.8:
            st.warning("💡 Advice: Adding Online Delivery or increasing Cuisines could improve this score.")
        else:
            st.success("🌟 Advice: Your setup is competitive! Focus on customer engagement.")

# --- MODE 2: CUSTOMER ---
else:
    st.title("🔍 Customer Discovery Portal")
    st.write("Find the best restaurants based on your budget and ratings.")
    
    budget = st.slider("Your Max Budget (USD)", 5, 200, 40)
    # Filter the data
    matches = df[df['Average_Cost_USD'] <= budget].sort_values(by="Aggregate rating", ascending=False)
    
    st.write(f"Showing top matches under ${budget}:")
    st.dataframe(matches[['Aggregate rating', 'Average_Cost_USD', 'Votes']].head(10))
    
    fig = px.scatter(matches.head(20), x="Average_Cost_USD", y="Aggregate rating", size="Votes", title="Price vs Rating Analysis")
    st.plotly_chart(fig)