import streamlit as st
import pickle 
import pandas as pd
import requests

movies_dict = pickle.load(open('model/movie_list.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('model/similarity.pkl', 'rb'))

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=37d01d4324ff8f203700d735cb7d6792&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    print(data)
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommend_movies = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommend_movies.append({
            'title': movies.iloc[i[0]].title,
            'poster': fetch_poster(movie_id)
        })
    return recommend_movies

st.title('Movie Recommender System')

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)
if st.button('Recommend'):
    recommended_movies = recommend(selected_movie)
    
    for movie in recommended_movies:
        st.text(movie['title'])
        st.image(movie['poster'])
