import streamlit as st
import pandas as pd
import numpy as np
from recommender import RecommendationEngine

# Configure page
st.set_page_config(
    page_title="Advanced Preference Elicitation Recommender",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load items data with caching for better performance."""
    try:
        items = pd.read_csv("data/items.csv")
        return items
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'data/items.csv' exists.")
        return pd.DataFrame()

@st.cache_resource
def initialize_engine(items_df):
    """Initialize recommendation engine with caching."""
    if items_df.empty:
        return None
    return RecommendationEngine(items_df)

def display_item_card(item):
    """Display a clean item card."""
    st.write(f"**{item['name']}**")
    
    if 'genre' in item and pd.notna(item['genre']):
        st.write(f"Genre: {item['genre']}")
    
    if 'year' in item and pd.notna(item['year']):
        st.write(f"Year: {int(item['year'])}")
    
    if 'rating' in item and pd.notna(item['rating']):
        # Convert 10-point ratings to 5-point scale
        rating_value = float(item['rating'])
        if rating_value > 5:
            rating_value = rating_value / 2
        st.write(f"Average: {rating_value:.1f}/5 ⭐")
    
    if 'description' in item and pd.notna(item['description']):
        st.write(f"*{item['description']}*")

def main():
    st.title("Movie, Food & Book Recommender")
    st.markdown("Rate items to get personalized recommendations!")
    
    # Load data and initialize engine
    items = load_data()
    if items.empty:
        st.stop()
    
    engine = initialize_engine(items)
    if engine is None:
        st.error("Failed to initialize recommendation engine.")
        st.stop()
    
    # Initialize session state
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        categories = sorted(items['category'].unique())
        selected_category = st.selectbox("Select Category:", categories)
        
        num_recommendations = st.slider("Number of recommendations:", 1, 10, 5)
        
        # Show your ratings
        actual_ratings = {k: v for k, v in st.session_state.user_ratings.items() if v > 0}
        if actual_ratings:
            st.subheader("Your Ratings")
            for item_name, rating in actual_ratings.items():
                stars = "⭐" * rating
                st.write(f"{item_name}: {stars} ({rating}/5)")
        
        if st.button("Clear All Ratings"):
            st.session_state.user_ratings = {}
            st.rerun()
    
    # Main content - TWO COLUMN LAYOUT ONLY
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header(f"Rate {selected_category} Items")
        
        # Filter items by category
        filtered_items = items[items['category'] == selected_category].copy()
        
        if len(filtered_items) == 0:
            st.warning(f"No items found in {selected_category} category.")
            return
        
        # Display items
        for idx, (_, item) in enumerate(filtered_items.iterrows()):
            item_name = item['name']
            current_rating = st.session_state.user_ratings.get(item_name, 0)
            
            with st.container():
                st.markdown("---")
                
                # Item details
                display_item_card(item)
                st.write("")
                
                # Rating interface
                if current_rating > 0:
                    # Show current rating clearly
                    stars_display = "⭐" * current_rating + "☆" * (5 - current_rating)
                    st.success(f"✅ Your rating: {stars_display} ({current_rating}/5)")
                
                # Rating buttons
                st.write("**Rate this item:**")
                cols = st.columns([1, 1, 1, 1, 1, 1])
                
                for i in range(1, 6):
                    with cols[i-1]:
                        if current_rating == i:
                            # Current rating - show as selected
                            if st.button(f"⭐ {i}", key=f"rate_left_{item_name}_{i}", use_container_width=True):
                                # Allow unrating by clicking same rating
                                del st.session_state.user_ratings[item_name]
                                st.rerun()
                        else:
                            # Not current rating
                            if st.button(f"☆ {i}", key=f"rate_left_{item_name}_{i}", use_container_width=True):
                                st.session_state.user_ratings[item_name] = i
                                st.rerun()
                
                # Clear button if rated
                if current_rating > 0:
                    with cols[5]:
                        if st.button("❌ Clear", key=f"clear_left_{item_name}", use_container_width=True):
                            del st.session_state.user_ratings[item_name]
                            st.rerun()
    
    with col2:
        st.header("Your Recommendations")
        
        # Generate recommendations
        if actual_ratings:
            try:
                recommendations = engine.get_recommendations(
                    user_ratings=actual_ratings,
                    category=str(selected_category),
                    num_recommendations=num_recommendations,
                    min_rating_threshold=3
                )
                
                if recommendations:
                    st.success(f"Found {len(recommendations)} recommendations!")
                    
                    for i, (item_name, score, explanation) in enumerate(recommendations):
                        with st.container():
                            st.markdown("---")
                            st.subheader(f"#{i+1} {item_name}")
                            
                            # Item details
                            item_data = items[items['name'] == item_name]
                            if not item_data.empty:
                                display_item_card(item_data.iloc[0])
                            
                            # Explanation
                            st.info(f"**Why recommended:** {explanation}")
                else:
                    st.warning(f"No recommendations available. Try rating more items.")
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
        else:
            st.info("👈 Rate some items to get personalized recommendations!")
            
            # Show popular items
            st.subheader("Popular Items in This Category")
            popular_items = filtered_items.copy()
            if 'rating' in popular_items.columns and len(popular_items) > 0:
                popular_items = popular_items[popular_items['rating'].notna()].sort_values('rating', ascending=False)
            
            for i, (_, item) in enumerate(popular_items.head(3).iterrows()):
                with st.container():
                    st.markdown("---")
                    display_item_card(item)
    
    # Footer stats
    st.markdown("---")
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.metric("Total Items", len(items))
    
    with col_stats2:
        st.metric("Items You've Rated", len(actual_ratings))
    
    with col_stats3:
        if actual_ratings:
            avg_rating = np.mean(list(actual_ratings.values()))
            st.metric("Your Average Rating", f"{avg_rating:.1f} ⭐")
        else:
            st.metric("Your Average Rating", "Not rated yet")

if __name__ == "__main__":
    main()