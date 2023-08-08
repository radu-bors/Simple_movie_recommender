import pandas as pd
import numpy as np

from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
import pickle

def recommend_popular(query,
                      ratings = pd.read_csv('../data/ml-latest-small/ratings.csv')
                      ):

    movies = pd.read_csv('../data/ml-latest-small/movies.csv')
    
    # which movies are in the query?
    movies.set_index('movieId').loc[query.keys()]
    
    # calculate the number of ratings per movie
    rating_per_movie = ratings.groupby('movieId')['userId'].count()
    rating_per_movie
    
    # filter for movies with more than 20 ratings and extract the index
    popular_movie = rating_per_movie.loc[rating_per_movie>20]
    popular_movie
    
    # filter the ratings matrix and only keep the popular movies
    ratings = ratings.set_index('movieId').loc[popular_movie.index]
    ratings = ratings.reset_index()
    
    # Initialize a sparse user-item rating matrix 
    R = csr_matrix((ratings['rating'], (ratings['userId'], ratings['movieId'])))
    
    # load previously trained model
    with open('./NMF_recommender.pkl', 'rb') as file:
        model = pickle.load(file)
        
    data=list(query.values())      # the ratings of the new user
    row_ind=[0]*len(data)          # we use just a single row 0 for this user
    col_ind=list(query.keys())
    
    # new user vector: needs to have the same format as the training data
    user_vec = csr_matrix((data, (row_ind, col_ind)), shape=(1, R.shape[1]))
    
    # user_vec -> encoding -> p_user_vec -> decoding -> user_vec_hat
    scores=model.inverse_transform(model.transform(user_vec))

    # convert to a pandas series
    scores = pd.Series(scores[0])
    
    # give a zero score to movies the user has allready seen
    scores[query.keys()] = 0
    
    # sort the scores from high to low 
    scores=scores.sort_values(ascending=False)
    
    # get the movieIds of the top 10 entries
    recommendations = scores.head(10).index

    return movies[movies['movieId'].isin(recommendations)]