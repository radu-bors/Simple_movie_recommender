import pandas as pd
import numpy as np
import random

from utils import MOVIES_LIST

from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
import pickle

def random_recommender(n=3):
    #chooses movies without replacement
    return random.sample(MOVIES_LIST, n)

def nmf_recommender(query,
                      ratings = pd.read_csv('../data/ml-latest-small/ratings.csv')
                      ):

    movies = pd.read_csv('../data/ml-latest-small/movies.csv')
    
    # Check if all movie IDs in the query are in the movies dataframe
    if not all(movie_id in movies['movieId'].values for movie_id in query.keys()):
        return ["Some of the movie IDs in the query are not found in the dataset."]
    
    # Calculate the number of ratings per movie
    rating_per_movie = ratings.groupby('movieId')['userId'].count()
    
    # Filter for movies with more than 20 ratings and extract the index
    popular_movie = rating_per_movie[rating_per_movie > 20]
    
    # Filter the ratings matrix and only keep the popular movies
    ratings = ratings.set_index('movieId').loc[popular_movie.index]
    ratings = ratings.reset_index()
    
    # Initialize a sparse user-item rating matrix 
    R = csr_matrix((ratings['rating'], (ratings['userId'], ratings['movieId'])))
    
    # Load previously trained model
    with open('./NMF_recommender.pkl', 'rb') as file:
        model = pickle.load(file)
        
    data = list(query.values())      # The ratings of the new user
    row_ind = [0] * len(data)        # We use just a single row 0 for this user
    col_ind = list(query.keys())
    
    # New user vector: needs to have the same format as the training data
    user_vec = csr_matrix((data, (row_ind, col_ind)), shape=(1, R.shape[1]))
    
    # user_vec -> encoding -> p_user_vec -> decoding -> user_vec_hat
    scores = model.inverse_transform(model.transform(user_vec))

    # Convert to a pandas series
    scores = pd.Series(scores[0])
    
    # Give a zero score to movies the user has already seen
    scores[query.keys()] = 0
    
    # Sort the scores from high to low 
    scores = scores.sort_values(ascending=False)
    
    # Get the movieIds of the top 10 entries
    recommendations = scores.head(10).index

    # Return the list of movie titles instead of the movies dataframe
    return list(movies[movies['movieId'].isin(recommendations)]['title'].values)