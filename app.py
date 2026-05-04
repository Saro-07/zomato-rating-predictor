import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go # Added for the visual Gauge Chart

# --- Setup & Configuration ---
st.set_page_config(page_title="Zomato Global AI Dashboard", layout="wide", page_icon="🍽️")

# 1. Caching mechanism properly retained to ensure fast asset loading
@st.cache_resource
def load_assets():
    """Loads the trained model, scaler, and cleaned dataset into cache."""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        df = pd.read_csv('cleaned_sample.csv')
        
        # Ensure categorical columns exist and handle NaNs for customer filtering
        if 'Cuisines' in df.columns:
            df['Cuisines'] = df['Cuisines'].fillna('Unknown')
        if 'City' in df.columns:
            df['City'] = df['City'].fillna('Unknown')
            
        return model, scaler, df
    except FileNotFoundError as e:
        st.error(f"Error loading assets: {e}. Please ensure model, scaler, and csv files are in the directory.")
        st.stop()

model, scaler, df = load_assets()

# Sidebar Optimization
st.sidebar.title("Select Your View")
st.sidebar.markdown("Navigate between predicting your own restaurant's success or finding top-rated food.")
mode = st.sidebar.radio("Choose Mode:", ["🏢 Business: Strategy & Simulation", "😋 Customer: Find My Food"])

# =====================================================================
# --- MODE 1: BUSINESS OWNER (Strategy & Simulation) ---
# =====================================================================
if mode == "🏢 Business: Strategy & Simulation":
    st.title("Business Strategy & Rating Predictor")
    st.info("Adjust the features below to simulate how operational changes impact your global Zomato rating.")

    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.subheader("Restaurant Configuration")
        
        # Feature 1: Cost (Consolidated to automatically calculate Price Category)
        cost = st.number_input(
            "Average Cost for Two (Standardized USD $)", 
            min_value=5, max_value=500, value=25, step=5,
            help="We use a unified USD baseline to compare your pricing against global standards."
        )
        
        # Auto-calculate Price Range based on standard Zomato logic to reduce user friction
        if cost < 10: price_range = 1
        elif cost < 25: price_range = 2
        elif cost < 50: price_range = 3
        else: price_range = 4

        # Feature 2: Popularity (Renamed to Target Votes to reflect it's a long-term goal)
        votes = st.slider(
            "Simulated Popularity (Target Votes)", 
            0, 5000, 100, step=50,
            help="Simulate the impact of gaining more customer reviews over time."
        )

        # Feature 3 & 4: Services
        delivery_choice = st.selectbox("Offer Online Delivery Service?", ["No", "Yes"])
        booking_choice = st.selectbox("Enable Table Reservations?", ["No", "Yes"])

        delivery = 1 if delivery_choice == "Yes" else 0
        booking = 1 if booking_choice == "Yes" else 0

        # Feature 5: Menu Complexity
        cuisines = st.slider("Menu Variety (Number of Cuisines)", 1, 10, 2)

    with col2:
        st.subheader("Model Prediction Analysis")
        
        # Prepare current simulation data
        feat = pd.DataFrame([[cost, price_range, votes, booking, delivery, cuisines]], 
                             columns=['Average_Cost_USD', 'Price range', 'Votes', 'Has Table booking', 'Has Online delivery', 'Cuisine_Count'])

        # Predict current rating
        pred = model.predict(scaler.transform(feat))[0]
        # Ensure prediction stays within valid Zomato rating bounds (0 to 5)
        pred = max(0.0, min(pred, 5.0))

        # Visual Upgrade: Gauge Chart instead of plain text metric
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Customer Rating", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#1E90FF"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 2.5], 'color': "#FF6347"},  # Red/Poor
                    {'range': [2.5, 4.0], 'color': "#FFD700"}, # Yellow/Average
                    {'range': [4.0, 5.0], 'color': "#32CD32"}  # Green/Excellent
                ],
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Dynamic "What-If" Feedback Logic
        st.markdown("### 📈 Dynamic Insights:")
        
        # Calculate Delta: What if they added delivery?
        if delivery == 0:
            feat_delivery = feat.copy()
            feat_delivery['Has Online delivery'] = 1
            pred_delivery = model.predict(scaler.transform(feat_delivery))[0]
            delta = pred_delivery - pred
            if delta > 0.05:
                st.warning(f"💡 **Service Tip:** Adding Online Delivery could boost your score by **+{delta:.2f} points**.")
            else:
                st.write("✔️ **Service Tip:** Online Delivery is recommended, but other factors are currently holding your score back more.")
        else:
            st.success("✔️ **Service Tip:** You are maximizing digital engagement by offering online delivery.")

        # Static/General Logic Tips
        if cuisines < 3:
            st.info("💡 **Menu Tip:** Increasing your cuisine variety may attract more diverse reviews.")
        if votes < 200:
            st.error("💡 **Marketing Tip:** Focus on getting your first 200 reviews to stabilize your rating. Your digital footprint is currently too low.")

# =====================================================================
# --- MODE 2: CUSTOMER (Discovery & Filtering) ---
# =====================================================================
else:
    st.title("Top Rated Restaurants Near You")
    st.write("Filter by your location, preferences, and budget to find the best-performing restaurants.")

    # Geo-Filtering & Preferences (Crucial UX Improvement)
    colA, colB, colC = st.columns(3)
    
    with colA:
        if 'City' in df.columns:
            cities = sorted(df['City'].unique().tolist())
            selected_city = st.selectbox("📍 Select City", ["All"] + cities)
        else:
            selected_city = "All"
            
    with colB:
        # Extract unique cuisines safely
        if 'Cuisines' in df.columns:
            all_cuisines = set(c.strip() for sublist in df['Cuisines'].dropna().str.split(',') for c in sublist)
            selected_cuisine = st.selectbox("🍝 Preferred Cuisine", ["All"] + sorted(list(all_cuisines)))
        else:
            selected_cuisine = "All"

    with colC:
        max_budget = st.number_input("💰 Max Budget for Two (USD $)", min_value=5, max_value=200, value=50, step=5)
        # Added a sorting preference
        sort_by = st.selectbox("Sort By", ["Highest Rating", "Most Popular (Votes)", "Lowest Price"])

    # Apply Filters
    matches = df.copy()
    matches = matches[matches['Average_Cost_USD'] <= max_budget]
    
    if selected_city != "All":
        matches = matches[matches['City'] == selected_city]
        
    if selected_cuisine != "All":
        matches = matches[matches['Cuisines'].str.contains(selected_cuisine, case=False, na=False)]

    # Apply Sorting
    if sort_by == "Highest Rating":
        matches = matches.sort_values(by=["Aggregate rating", "Votes"], ascending=[False, False])
    elif sort_by == "Most Popular (Votes)":
        matches = matches.sort_values(by="Votes", ascending=False)
    else:
        matches = matches.sort_values(by="Average_Cost_USD", ascending=True)

    # Format Display
    if matches.empty:
        st.warning("No restaurants found matching your criteria. Try expanding your budget or changing the city.")
    else:
        st.subheader(f"Top Recommended Restaurants (Found {len(matches)} matches)")
        
        # Displaying the top 15 results beautifully using st.dataframe configuration
        matches['Price (USD)'] = matches['Average_Cost_USD'].apply(lambda x: f"${x:.2f}")
        
        display_cols = ['Restaurant Name', 'City', 'Locality', 'Cuisines', 'Aggregate rating', 'Price (USD)', 'Votes']
        
        # Ensure only columns that actually exist in the dataframe are displayed
        display_cols = [col for col in display_cols if col in matches.columns]
        
        st.dataframe(
            matches[display_cols].head(15), 
            use_container_width=True,
            hide_index=True # Hides the arbitrary pandas index numbers for a cleaner look
        )
        
        st.success("Results updated based on global data trends and your personal filters.")