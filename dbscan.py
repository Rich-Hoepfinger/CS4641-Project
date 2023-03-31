import numpy as np
class DBSCAN(object):
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset

    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        """
        cluster_idx = np.full(len(self.dataset), -1)
        visitedIndices = set()
        C = 0
        
        for pIndex in range(len(self.dataset)):
            p = self.dataset[pIndex]
            if (not (pIndex in visitedIndices)):
                visitedIndices.add(pIndex)
                neighborPoints = self.regionQuery(pIndex)
                if (len(neighborPoints) < self.minPts):
                    cluster_idx[pIndex] = -1
                else:
                    self.expandCluster(pIndex, neighborPoints, C, cluster_idx, visitedIndices)
                    C += 1
        
        return cluster_idx

    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        """
        cluster_idx[index] = C
        while (len(neighborIndices) > 0):
            neighborIndices = np.unique(np.sort(neighborIndices))
            current = neighborIndices[0]
            neighborIndices = np.take(neighborIndices, np.arange(1, len(neighborIndices), dtype=np.int_))
            if (not (current in visitedIndices)):
                visitedIndices.add(current)
                newNeighbors = self.regionQuery(current)
                if len(newNeighbors) >= self.minPts:
                    neighborIndices = np.concatenate((neighborIndices, newNeighbors), dtype=np.int_)
            if (cluster_idx[current] < 0):
                cluster_idx[current] = C
                

    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        """
        return np.nonzero(self.pairwise_dist(self.dataset, self.dataset[pointIndex][np.newaxis, :])
 < self.eps)[0]
    
    def pairwise_dist(self, x, y):
        """Returns the norms of every point in x against y
        
        Args:
			x: M x D numpy array
			y: 1 x D numpy array
        Return:
			Dist: M x 1 numpy array of the distance of each point in x from y 
        """
        return np.linalg.norm(x - y, axis=1)