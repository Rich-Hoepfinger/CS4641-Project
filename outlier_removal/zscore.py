import pandas as pd
import numpy as np


def min_max(logit):
    '''
    Args:
        logit: N x D pd.DataFrame
    Return:
        score: N x D pd.DataFrame. A normalised score in strict scale 0-1 for each dimension.
    '''
    logit = (logit - logit.min(axis=0))/(logit.max(axis=0) - logit.min(axis=0))
    return logit

def zscore(logit):
    '''
    Args:
        logit: N x D pd.DataFrame
    Return:
        score: N x D pd.DataFrame. A standardised score for each dimension.
    '''
    # Datapoints 1 standard deviation awasy mean will be < 0 or > 1. 
    logit = (logit - logit.mean(axis=0))/logit.std(axis=0, ddof = 1)
    return logit

def softmax(logit):  # [5pts]
    """
    Args:
        logit: N x D numpy array
    Return:
        prob: N x D numpy array. A probability distribution over the set of dimensions. 
    """
    logit = logit - logit.max(axis=1).reshape(len(logit), 1)
    sum_exp = np.sum(np.exp(logit), axis = 1, keepdims = True)
    prob = np.exp(logit) / sum_exp
    return prob

def zscore_main():
    data = pd.read_csv("outlier_removal/automated_dbscanned_trackdata.csv", sep = ",")
    normalisation = data[['popularity', 'loudness', 'tempo', 'duration_ms']]
    data_minmax = min_max(normalisation)
    data[['popularity', 'loudness', 'tempo', 'duration_ms']] = data_minmax
    data.to_csv("outlier_removal/minmax_normalised_trackdata.csv", index = False)

    # Change every numeric variable to zscore normalised.
    index_set = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness','instrumentalness', 'liveness', 
                'valence', 'tempo', 'duration_ms']
    data[index_set] = zscore(data[index_set])
    data.to_csv("outlier_removal/automated_zscore_normalised_trackdata.csv", index = False)
    
zscore_main()