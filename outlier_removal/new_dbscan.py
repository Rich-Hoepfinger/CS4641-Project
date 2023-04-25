import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
#dupremoved -> dbscan -> zscore -> collinearity -> k nearest neighbors

def dbscan_main(eps, min_points, metric):
    org_data = pd.read_csv("dupremoved_trackdata.csv", sep = ",")

    # Column 9, "mode", has been removed to increase sanity of results, as it restricted DBScan to only 2 possible clusters
    # put as '8' back into the hardcoded array of numbers to add it back in
    # Column 10, time was milliseconds & has been removed for obvious reasons
    data = org_data.to_numpy()[:, [3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 17]].astype(np.double)
    

    scanner = DBSCAN(eps=eps, min_samples=min_points, metric=metric).fit(data)
    labels = scanner.labels_
    bi = [bool(i) for i in (labels + 1)]
    bool_idx = np.array(bi)

    cleaned_data = org_data.to_numpy()[bool_idx]
    cleaned_dataframe = pd.DataFrame(data=cleaned_data, columns=org_data.columns, index=org_data.index[bool_idx])
    cleaned_dataframe.to_csv("outlier_removal/automated_dbscanned_trackdata.csv", sep=",")
    
    removed_songs = np.shape(org_data)[0]-np.shape(cleaned_data)[0]
    
    with open('outlier_removal/tests.txt','a') as f:
        f.write("\nRemoved Songs: " + str(removed_songs))
        f.write("\nEpsilon: " + str(eps))
        f.write("\nMin_Points: " + str(min_points))
        f.write("\nMetric: " + str(metric))
    
    return removed_songs

dbscan_main(eps=15, min_points=3,metric="cityblock")
    