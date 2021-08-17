from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.spatial.distance import cdist

def compute_bic(centers, labels, X, m):
    n = np.bincount(labels,minlength=m)
    N, d = X.shape
    cl_var = (1.0 / (N - m) / d) * sum([sum(cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term
    return BIC
    

def compute_cluster_mapping( c1, c2, n_clusters):
        """
        Compute cost matrix to map cluster assigments c1 -> cluster assignments c2.
        The cost of mapping label i -> label j is equal to |c1(i)| + |c2(j)| -2*|c1(i) intersection c2(j)| 
        where c1(i) is the set of indexes where c1==i 
        and c2(j) is the set of indexes where c2==j
        Args:
            c1 (np.ndarray): cluster assigment #1 with shape (n_samples, )
            c2 (np.ndarray): cluster assignment #2 with shape (n_samples, )
            
        Returns:
            C (np.ndarray): Cost Matrix (n_clusters, n_clusters) 
        """

        C = np.zeros((n_clusters,n_clusters))
        for i in range(n_clusters):
            c1_i = np.nonzero(c1==i)[0]
            for j in range(n_clusters):
                c2_j = np.nonzero(c2==j)[0]
                C[i][j] = c1_i.size + c2_j.size - 2*np.intersect1d(c1_i,c2_j).size
        
        row_ind, col_ind = linear_sum_assignment(C)
        return col_ind