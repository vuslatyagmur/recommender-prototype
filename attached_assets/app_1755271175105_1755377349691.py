import streamlit as st
import pandas as pd

items = pd.read_csv("data/items.csv")

st.title("Preference Elicitation Recommender")

categories = items['category'].unique()
selected_category = st.selectbox("Which category are you interested in?", categories)

filtered_items = items[items['category'] == selected_category]

ratings = {}
for idx, row in filtered_items.iterrows():
    ratings[row['name']] = st.slider(f"Rate {row['name']}", 1, 5, 3)

if st.button("Get Recommendation"):
    recommended = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    st.subheader("Recommended item:")
    st.write(recommended[0][0])
