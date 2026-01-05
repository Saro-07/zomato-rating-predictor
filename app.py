import streamlit as st
import pandas as pd
import joblib

# Setup
st.set_page_config(page_title="Zomato Global AI", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv('cleaned_sample.csv')
    return model, scaler, df

model, scaler, df = load_assets()

st.sidebar.title("Select Your View")
mode = st.sidebar.radio("Choose Mode:", ["🏢 Business: Strategy & Simulation", "😋 Customer: Find My Food"])

# --- MODE 1: BUSINESS OWNER (Clearer Features) ---
if mode == "🏢 Business: Strategy & Simulation":
    st.title("Business Strategy & Rating Predictor")
    st.info("Adjust the features below to see how they impact your predicted restaurant rating.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Restaurant Configuration")
        cost = st.number_input("Average Cost for Two (Standardized USD $)", 5, 500, 25)
        votes = st.slider("Public Popularity (Total Votes)", 0, 5000, 100)
        
        # Human-friendly labels for features
        delivery_choice = st.selectbox("Offer Online Delivery Service?", ["No", "Yes"])
        booking_choice = st.selectbox("Enable Table Reservations?", ["No", "Yes"])
        
        delivery = 1 if delivery_choice == "Yes" else 0
        booking = 1 if booking_choice == "Yes" else 0
        
        cuisines = st.slider("Menu Variety (Number of Cuisines)", 1, 10, 2)
        price_range = st.select_slider("Price Category (1: Budget, 4: Luxury)", options=[1, 2, 3, 4])

    with col2:
        st.subheader("Model Prediction Analysis")
        # Prepare data exactly as the 5 models were trained
        feat = pd.DataFrame([[cost, price_range, votes, booking, delivery, cuisines]], 
                            columns=['Average_Cost_USD', 'Price range', 'Votes', 'Has Table booking', 'Has Online delivery', 'Cuisine_Count'])
        
        pred = model.predict(scaler.transform(feat))[0]
        st.metric("Predicted Customer Rating", f"{pred:.2f} / 5.0")
        
        # Simplified Business Explanation
        st.markdown("---")
        st.markdown("### 📈 How to Improve your Score:")
        if delivery == 0:
            st.write("- **Service Tip:** Adding Online Delivery is a major rating driver.")
        if cuisines < 3:
            st.write("- **Menu Tip:** Increasing your cuisine variety may attract more diverse reviews.")
        if votes < 200:
            st.write("- **Marketing Tip:** Focus on getting your first 200 reviews to stabilize your rating.")

# --- MODE 2: CUSTOMER (Names & Locations) ---
else:
    st.title("Top Rated Restaurants Near You")
    st.write("Filter by your budget to find the best-performing locations.")
    
    max_budget = st.slider("Your Max Budget (USD $)", 5, 200, 50)
    
    # Filtering the data and showing real names/locations
    matches = df[df['Average_Cost_USD'] <= max_budget].copy()
    
    # Sorting to show 'Best' first
    matches = matches.sort_values(by="Aggregate rating", ascending=False)
    
    # Formatting the Cost column for the display table
    matches['Price (USD)'] = matches['Average_Cost_USD'].apply(lambda x: f"${x:.2f}")
    
    st.subheader(f"Top Recommended Restaurants (Under ${max_budget})")
    
    # Showing the actual Name and Location
    display_cols = ['Restaurant Name', 'City', 'Locality', 'Aggregate rating', 'Price (USD)', 'Votes']
    st.dataframe(matches[display_cols].head(15), use_container_width=True)

    st.success("Results are sorted by the highest predicted rating based on global data trends.")