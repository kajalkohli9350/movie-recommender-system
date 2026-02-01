
import pandas as pd
import numpy as np
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import streamlit as st
import requests
import os
# Load data
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")


# Merge on movie id
movies = movies.merge(
    credits,
  on='title'
)

# Select relevant features
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop missing values
movies.dropna(inplace=True)

#  define a function to Convert stringified lists to actual lists
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])

    return L

# Apply the function to genres and keywords columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Define a function to get the top 3 cast members
def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

# Apply the function to cast column
movies['cast'] = movies['cast'].apply(convert_cast)

# Define a function to get the director from crew
def fectch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# Apply the function to crew column
movies['crew'] = movies['crew'].apply(fectch_director)

# Split the overview into list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces in multi-word features
def remove_space(word):
    L = []
    for i in word:
        L.append(i.replace(" ", ""))
    return L
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)
movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)

# Create a new feature 'tags' by combining relevant features
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a new dataframe with only 'movie_id', 'title', and 'tags'
new_df = movies[['movie_id', 'title', 'tags']]

# Convert list of tags to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Convert tags to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Create Count Vectorizer object
ps = PorterStemmer()
def stem(text):
    l = []
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)

# Apply stemming to tags
new_df['tags'] = new_df['tags'].apply(stem)

# Create Count Vectorizer and transform tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Compute cosine similarity
similarity = cosine_similarity(vectors)

# Function to recommend movies
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)
    for i in movies_list[1:6]:
        print(new_df.iloc[i[0]].title)
# Example usage
print(recommend('The Dark Knight Rises'))


if not os.path.exists("artifacts"):
    os.mkdir("artifacts")

# Save files
pickle.dump(new_df, open("artifacts/movie_list.pkl", "wb"))
pickle.dump(similarity, open("artifacts/similarity.pkl", "wb"))
print("Artifacts created successfully ✅")

# Function to fetch poster from TMDB API


    

# Function to recommend movies
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)

    recommended_movies_name = []
   
    for i in movies_list[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
       
        recommended_movies_name.append(movies.iloc[i[0]].title)
    return recommended_movies_name

#streamlit app 
st.header('Movie Recommender System')
movies= pickle.load(open('artifacts/movie_list.pkl','rb'))
similarity= pickle.load(open('artifacts/similarity.pkl','rb'))

# Selectbox for movie selection
movie_list= movies['title'].values
selected_movie= st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)
if st.button('Show Recommendation'):
    recommended_movies_name= recommend(selected_movie)
    col1, col2, col3, col4, col5= st.columns(5)
    with col1:
        st.text(recommended_movies_name[0])
        
    with col2:
        st.text(recommended_movies_name[1])
        
    with col3:
        st.text(recommended_movies_name[2])
    with col4:
        st.text(recommended_movies_name[3])
    with col5:
        st.text(recommended_movies_name[4])

