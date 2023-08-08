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

def nmf_recommender():
    pass

if __name__ == '__main__':
    print(f"Your 3 recommendations are:\n{', '.join(random_recommender())}")