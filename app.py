import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Dashboard Title and Description
st.title("🍴 Global Restaurant Rating Predictor")
st.write("""
This dashboard uses a **Stacked Ensemble Machine Learning Model** to predict a restaurant's aggregate rating 
based on global Zomato data.
""")

st.sidebar.header("Enter Restaurant Details")

# 3. User Inputs in the Sidebar
def user_input_features():
    avg_cost = st.sidebar.number_input("Average Cost for Two (USD)", min_value=1.0, max_value=1000.0, value=25.0)
    price_range = st.sidebar.slider("Price Range (1 = Cheap, 4 = Luxury)", 1, 4, 2)
    votes = st.sidebar.number_input("Number of Votes (Popularity)", min_value=0, max_value=20000, value=100)
    
    table_booking = st.sidebar.selectbox("Has Table Booking?", ["Yes", "No"])
    online_delivery = st.sidebar.selectbox("Has Online Delivery?", ["Yes", "No"])
    cuisine_count = st.sidebar.slider("Number of Cuisines Offered", 1, 8, 3)

    # Convert Yes/No to 1/0
    data = {
        'Average_Cost_USD': avg_cost,
        'Price range': price_range,
        'Votes': votes,
        'Has Table booking': 1 if table_booking == "Yes" else 0,
        'Has Online delivery': 1 if online_delivery == "Yes" else 0,
        'Cuisine_Count': cuisine_count
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Display the inputs
st.subheader("Restaurant Profile")
st.write(input_df)

# 5. Prediction Logic
if st.button("Predict Rating"):
    # Scale the input just like we did in training
    scaled_input = scaler.transform(input_df)
    
    # Get prediction
    prediction = model.predict(scaled_input)
    
    # Display Result
    st.success(f"### Predicted Aggregate Rating: {prediction[0]:.2f} / 5.0")
    
    # Add a humanized tip
    if prediction[0] >= 4.0:
        st.balloons()
        st.write("🌟 **Analysis:** This restaurant is likely to be a top-performer in the global market!")
    elif prediction[0] >= 3.0:
        st.write("👍 **Analysis:** This is an average performing restaurant with steady customer satisfaction.")
    else:
        st.write("⚠️ **Analysis:** This configuration might lead to lower customer satisfaction. Consider adding services like Online Delivery.")

st.info("Note: This model was trained on global standardized data using USD conversion.")
