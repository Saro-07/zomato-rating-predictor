import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- SETUP ---
st.set_page_config(page_title="Zomato Analytics", layout="wide")

@st.cache_resource
def load_data():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    # Load a small sample of your cleaned data for the charts
    df = pd.read_csv('cleaned_sample.csv') 
    return model, scaler, df

model, scaler, df = load_data()

# --- SIDEBAR: INPUT PANEL ---
st.sidebar.header("📍 Restaurant Configuration")
country = st.sidebar.selectbox("Select Country Market", ["India", "USA", "UAE", "UK", "Others"])
local_price = st.sidebar.number_input(f"Cost in Local Currency", value=1000)

# Working Function: The Global Scaler
rates = {"India": 0.012, "USA": 1.0, "UAE": 0.27, "UK": 1.27, "Others": 1.0}
usd_cost = local_price * rates[country]

st.sidebar.markdown(f"**Standardized Cost:** ${usd_cost:.2f} USD")

votes = st.sidebar.slider("Current Votes (Popularity)", 0, 5000, 150)
delivery = st.sidebar.checkbox("Offers Online Delivery")
booking = st.sidebar.checkbox("Offers Table Booking")
cuisines = st.sidebar.slider("Cuisine Count", 1, 8, 2)
price_range = st.sidebar.select_slider("Price Tier", options=[1, 2, 3, 4])

# --- MAIN CONTENT ---
st.title("🍴 Global Restaurant Strategy Dashboard")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("🎯 Prediction Engine")
    
    # Prepare Input
    input_data = pd.DataFrame({
        'Average_Cost_USD': [usd_cost],
        'Price range': [price_range],
        'Votes': [votes],
        'Has Table booking': [1 if booking else 0],
        'Has Online delivery': [1 if delivery else 0],
        'Cuisine_Count': [cuisines]
    })
    
    # Predict
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    
    # Display Result
    st.metric(label="Predicted Aggregate Rating", value=f"{prediction:.2f} / 5.0")
    
    # Working Function: Improvement Advisor
    st.subheader("💡 Strategic Recommendations")
    if not delivery:
        st.info("🚀 **Growth Tip:** Our models show that adding **Online Delivery** could significantly boost your visibility and rating.")
    if votes < 500:
        st.warning("📈 **Marketing Tip:** Your vote count is low. Encourage customer reviews to increase 'Social Proof'.")
    if prediction >= 4.0:
        st.balloons()
        st.success("Your restaurant is currently in the 'Top Performer' bracket!")

with col2:
    st.header("📊 Market Context")
    # Working Function: Comparative Analytics
    fig = px.scatter(df, x="Votes", y="Aggregate rating", color="Price range",
                     title="Where do you stand? (Your Inputs vs. Market)")
    # Add a gold star for the user's prediction
    fig.add_scatter(x=[votes], y=[prediction], mode='markers', 
                    marker=dict(color='Gold', size=15, symbol='star'), name='Your Prediction')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Powered by Stacked Ensemble Learning | Standardized Global Dataset")