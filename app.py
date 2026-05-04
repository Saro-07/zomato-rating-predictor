import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# --- Setup & Configuration ---
st.set_page_config(page_title="Zomato Global AI | Enterprise", layout="wide", page_icon="🍽️")

# --- Initialize Session State for Customer Favorites ---
if 'shortlist' not in st.session_state:
    st.session_state.shortlist = []

@st.cache_resource
def load_assets():
    """Loads the trained model, scaler, and the FULL dataset into cache."""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # 🚨 IMPORTANT: Change this to the exact name of your FULL dataset file (the one with 7,403 rows)
        df = pd.read_csv('cleaned_sample.csv', encoding='latin-1') 
        
        # Data Safety: Ensure Cuisines and Cities are strings and handle missing data
        if 'Cuisines' in df.columns:
            df['Cuisines'] = df['Cuisines'].fillna('Unknown').astype(str)
        if 'City' in df.columns:
            df['City'] = df['City'].fillna('Unknown').astype(str)
            
        return model, scaler, df
    except FileNotFoundError as e:
        st.error(f"Error loading assets: {e}. Please ensure your full CSV dataset is in the folder.")
        st.stop()

model, scaler, df = load_assets()

# --- Sidebar Navigation ---
st.sidebar.title("Zomato Enterprise AI")
st.sidebar.markdown("Select your workspace environment.")
mode = st.sidebar.radio("Workspace:", ["🏢 Business: Strategy & Simulation", "😋 Customer: Discovery Hub"])

# =====================================================================
# --- MODE 1: BUSINESS OWNER (Strategy & Simulation) ---
# =====================================================================
if mode == "🏢 Business: Strategy & Simulation":
    st.title("Business Strategy & Predictive Analytics")
    st.info("Simulate operational changes, forecast ratings, and measure estimated market impact.")

    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.subheader("Operational Parameters")
        
        cost = st.number_input("Average Cost for Two (USD $)", min_value=1, max_value=1000, value=25, step=5)
        
        # Auto-calculate Price Range
        if cost < 10: price_range = 1
        elif cost < 25: price_range = 2
        elif cost < 50: price_range = 3
        else: price_range = 4

        votes = st.slider("Target Popularity (Total Votes)", 0, 5000, 100, step=50)
        delivery_choice = st.selectbox("Offer Online Delivery?", ["No", "Yes"])
        booking_choice = st.selectbox("Enable Table Reservations?", ["No", "Yes"])

        delivery = 1 if delivery_choice == "Yes" else 0
        booking = 1 if booking_choice == "Yes" else 0
        cuisines = st.slider("Menu Diversity (Cuisine Count)", 1, 15, 2)

    with col2:
        st.subheader("Predictive Market Outcome")
        
        feat = pd.DataFrame([[cost, price_range, votes, booking, delivery, cuisines]], 
                             columns=['Average_Cost_USD', 'Price range', 'Votes', 'Has Table booking', 'Has Online delivery', 'Cuisine_Count'])

        pred = model.predict(scaler.transform(feat))[0]
        pred = max(0.0, min(pred, 5.0))

        # Enterprise Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = pred,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Forecasted Rating", 'font': {'size': 18}},
            delta = {'reference': 3.5, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}}, 
            number = {'font': {'size': 45}},
            gauge = {
                'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)", 'thickness': 0},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 2.5], 'color': "#EF5350"},
                    {'range': [2.5, 4.0], 'color': "#FFCA28"},
                    {'range': [4.0, 5.0], 'color': "#66BB6A"}
                ],
                'threshold': {'line': {'color': "black", 'width': 6}, 'thickness': 0.75, 'value': pred}
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # Advanced ROI / Market Impact
        st.markdown("### 📊 Market Intelligence & ROI")
        
        tab1, tab2 = st.tabs(["Actionable Insights", "Competitive Benchmark"])
        
        with tab1:
            if delivery == 0:
                feat_delivery = feat.copy()
                feat_delivery['Has Online delivery'] = 1
                pred_delivery = max(0.0, min(model.predict(scaler.transform(feat_delivery))[0], 5.0))
                delta = pred_delivery - pred
                if delta > 0.05:
                    rev_impact = delta * 15 
                    st.success(f"**Growth Opportunity:** Adding Online Delivery boosts predicted rating by **+{delta:.2f}**. Industry data suggests this could drive a **~{rev_impact:.1f}% increase** in digital order volume.")
            else:
                st.info("✔️ Digital footprint optimized. Delivery services are active.")

            if votes < 200:
                st.error("⚠️ **Risk Alert:** Restaurants with under 200 votes suffer from high rating volatility. Invest in social proof marketing.")

            # Export Configuration feature
            csv_export = feat.copy()
            csv_export['Forecasted_Rating'] = round(pred, 2)
            csv = csv_export.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Strategy Report (CSV)", csv, "restaurant_strategy_forecast.csv", "text/csv")

        with tab2:
            st.write("Compare your simulated rating against the global average for your price bracket.")
            if 'Price range' in df.columns and 'Aggregate rating' in df.columns:
                market_avg = df[df['Price range'] == price_range]['Aggregate rating'].mean()
                if pd.notna(market_avg):
                    diff = pred - market_avg
                    st.metric(label=f"Market Average (Price Tier {price_range})", value=f"{market_avg:.2f}", delta=f"{diff:.2f} vs Market")
                else:
                    st.write("Insufficient market data for this specific configuration.")

# =====================================================================
# --- MODE 2: CUSTOMER (Discovery Hub) ---
# =====================================================================
else:
    st.title("Global Discovery Hub")
    st.write("Curate your dining experience with advanced market filtering.")

    colA, colB, colC = st.columns(3, gap="medium")
    
    with colA:
        # 🚨 FIX APPLIED: Extracting ALL valid cities dynamically
        if 'City' in df.columns:
            # Filter out blanks and Unknowns, then sort alphabetically
            valid_cities = [city.strip() for city in df['City'].unique() if city.strip() != '' and city != 'Unknown']
            cities = sorted(valid_cities)
            selected_city = st.selectbox("📍 Global Location", ["All"] + cities, index=cities.index("Chennai") + 1 if "Chennai" in cities else 0)
        else:
            selected_city = "All"
            
        if 'Cuisines' in df.columns:
            all_cuisines = set(c.strip() for sublist in df['Cuisines'].dropna().str.split(',') for c in sublist if c.strip() != '')
            selected_cuisine = st.selectbox("🍝 Cuisine Focus", ["All"] + sorted(list(all_cuisines)))
        else:
            selected_cuisine = "All"

    with colB:
        max_budget = st.number_input("💰 Max Budget for Two (USD $)", min_value=5, max_value=500, value=60, step=5)
        min_rating = st.slider("⭐ Minimum Rating Quality", 0.0, 5.0, 4.0, step=0.1)

    with colC:
        sort_by = st.selectbox("Sort Priority", ["Highest Quality (Rating)", "Highest Popularity (Votes)", "Best Value (Price)"])
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear Personal Shortlist"):
            st.session_state.shortlist = []

    # --- Apply Enterprise Filters ---
    matches = df.copy()
    if 'Average_Cost_USD' in matches.columns:
        matches = matches[matches['Average_Cost_USD'] <= max_budget]
    if 'Aggregate rating' in matches.columns:
        matches = matches[matches['Aggregate rating'] >= min_rating]
    if selected_city != "All":
        matches = matches[matches['City'] == selected_city]
    if selected_cuisine != "All":
        matches = matches[matches['Cuisines'].str.contains(selected_cuisine, case=False, na=False)]

    # --- Apply Sorting ---
    if sort_by == "Highest Quality (Rating)":
        matches = matches.sort_values(by=["Aggregate rating", "Votes"], ascending=[False, False])
    elif sort_by == "Highest Popularity (Votes)":
        matches = matches.sort_values(by="Votes", ascending=False)
    else:
        if 'Average_Cost_USD' in matches.columns:
            matches = matches.sort_values(by="Average_Cost_USD", ascending=True)

    if matches.empty:
        st.warning("No restaurants match this strict criteria. Try lowering the minimum rating or increasing the budget.")
    else:
        st.markdown(f"**Market Snapshot:** {len(matches)} options found. Average Cost: ${matches['Average_Cost_USD'].mean():.2f}")
        
        matches['Price (USD)'] = matches['Average_Cost_USD'].apply(lambda x: f"${x:.2f}")
        if 'Has Online delivery' in matches.columns:
            matches['Delivery'] = matches['Has Online delivery'].map({1: 'Yes', 0: 'No', 'Yes': 'Yes', 'No':'No'})
        
        display_cols = ['Restaurant Name', 'City', 'Locality', 'Cuisines', 'Aggregate rating', 'Price (USD)', 'Votes', 'Delivery']
        display_cols = [col for col in display_cols if col in matches.columns]
        
        interactive_df = matches[display_cols].copy()
        interactive_df.insert(0, "Shortlist", False) 

        st.subheader("Market Results")
        edited_df = st.data_editor(
            interactive_df,
            hide_index=True,
            use_container_width=True,
            height=350,
            disabled=display_cols 
        )

        new_favorites = edited_df[edited_df["Shortlist"] == True]["Restaurant Name"].tolist()
        if new_favorites:
            for fav in new_favorites:
                if fav not in st.session_state.shortlist:
                    st.session_state.shortlist.append(fav)

        if st.session_state.shortlist:
            with st.expander("⭐ Your Personalized Dining Shortlist", expanded=True):
                for item in set(st.session_state.shortlist):
                    st.markdown(f"- **{item}**")