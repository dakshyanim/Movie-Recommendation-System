import streamlit as st
import pickle
import numpy as np
import pandas as pd
import io

# Load movie list and similarity matrix
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title(" Movie Recommendation System")
st.image("https://images.pexels.com/photos/5662857/pexels-photo-5662857.png", width=400) 
st.write("Get top movie recommendations based on your favorite movie!")

# Dropdown list of movie titles
movie_list = movies['title'].values
selected_movie = st.selectbox("Choose a movie", movie_list)

# Function to recommend movies 
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )
    recommended_movies = []
    for i in distances[1:31]:  
        title = movies.iloc[i[0]].title
        score = similarity[index][i[0]]
        recommended_movies.append((title, round(score, 3)))  # Return tuple
    return recommended_movies

#  Button to show recommendations
if st.button("Show Recommendations"):
    st.subheader(f" Recommended movies similar to **{selected_movie}**:")
    recommendations = recommend(selected_movie)
    for i, (movie, score) in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}  (Similarity Score: {score})")

#  Button to download recommendations as Excel
if st.button("Dowload Recommendations"):
    recommendations = recommend(selected_movie)
    df = pd.DataFrame(recommendations, columns=['Recommended Movies', 'Similarity Score'])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Recommendations')
    output.seek(0)

    st.download_button(
        label=" Download Recommended Movies (Excel)",
        data=output,
        file_name=f"recommendations_for_{selected_movie}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

