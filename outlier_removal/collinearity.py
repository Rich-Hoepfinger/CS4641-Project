import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os

#dupremoved -> dbscan -> zscore -> collinearity -> k nearest neighbors

def remove_collinearity(threshold):
    # os.chdir('/model')
    df = pd.read_csv('outlier_removal/automated_zscore_normalised_trackdata.csv')
    # os.chdir('../outlier_removal')
    finished = False
    df = df.iloc[:,4:].drop('id', axis = 1)
    features = df.iloc[:,:-1]
    dropped_columns = []
    while(not finished):
        corrvals = features.corr()
        removed = False
        for (i, column_name) in enumerate(corrvals):
            for j in range(i):
                if i != j:
                    if corrvals.iloc[i,j] > threshold or corrvals.iloc[i, j] < -1 * threshold:
                        features.drop(column_name, axis = 1, inplace=True)
                        dropped_columns.append(column_name)
                        removed = True
                        break
        if removed is False:
            finished = True
            break

    df.drop(dropped_columns, axis = 1, inplace = True)
    with open('outlier_removal/tests.txt','a') as f:
        f.write("\nColinearity Threashold: " + str(threshold))
        f.write("\nDropped Columns: " + str(dropped_columns))
        
    df.to_csv("outlier_removal/automated_collinearity_removed.csv", index = False)

remove_collinearity(0.8)