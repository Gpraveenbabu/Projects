#importing the required
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#loading the data from the csv file to apandas dataframe

movies_data = pd.read_csv('/content/movies.csv')
movies_data.head()
#selecting the relevant features for recommendation

selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)
# replacing the null valuess with null string

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')
# combinig the features we are considered

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
print(combined_features)
#converting the text data to feature vector

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
print(feature_vectors)
#for getting the similarities using cosine similarity

similarity = cosine_similarity(feature_vectors)
print(similarity)
#reading the movie name input from the user

movie_name = input('Enter the name of your favourite movie name:')
# finding the close match from the dataset which is relavent to the user input

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)
#taking close match from relative list

close_match = find_close_match[0]
print(close_match)
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)
#for testing the similarty score from the relative movies

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)
len(similarity_score)
