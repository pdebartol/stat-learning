import sklearn
import numpy as np
from sklearn.neighbors import DistanceMetric

class SmoothKMeans(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Template class for Smooth KMeans

    Attributes:
        cluster_centers (np.ndarray): Cluster centroids y_i
            (n_clusters, n_features)
        cluster_probs (np.ndarray): Assignment probability vectors
            q(y_i | x) for each sample (n_samples, n_clusters)
            
    Parameters:
        n_clusters (int): Maximum number of clusters returned by DAC.
        random_state (int): Random seed.
    """

    def __init__(self, n_clusters=3, convergence_em=1e-15,
                 metric="euclidean", threshold=3.02, T=1e-1, lambda_param=1e-1, random_seed=40):
        self.n_cluster = n_clusters
        self.metric = metric
        self.convergence_em = convergence_em
        self.cluster_centers = None
        self.cluster_probs = None
        self.mean_fields = None
        self.threshold = threshold
        self.temperature = T
        self.lambda_param = lambda_param
        np.random.seed(random_seed)

        

    def fit(self, X):
        """
        Compute SmoothKMeans for feature matrix X

        Args:
        X (np.ndarray): Input array with shape (n_x, n_features)
        """
        #  1) Initialize
        n_x = X.shape[0]
        n_features = X.shape[1]
        epsilon = np.random.normal(scale=1e-1,loc=1, size=(self.n_cluster, n_features))
        self.cluster_centers = np.tile(np.mean(X, axis=0),(self.n_cluster,1)) + epsilon  # centroids = sample_mean
        self.cluster_probs = np.random.dirichlet([1/self.n_cluster for _ in range(self.n_cluster)], size=n_x)
        self.mean_fields = np.random.rand(n_x, self.n_cluster)
        # 2) Compute Neighborhoods
        N = self.get_neighborhood(X) 

        # 2) Repeat until convergence
        
        while True: 
            em_delta = self._em_step(X, N) # EM algorithm
            
            while em_delta > self.convergence_em:
                em_delta = self._em_step(X, N) 
            break
        

    def _em_step(self,X, D):
        n_x = X.shape[0]
        n_features = X.shape[1]
        
        # pick a random visit schedule for u
        us = np.random.permutation(range(n_x))
        
        for u in us:
            # \forall alpha: || x_u - y_alpha ||^2 
            x = np.tile(X[u,:],(self.n_cluster,1))
            cost_1 = np.square(np.linalg.norm(x-self.cluster_centers, axis=1))
            # \forall j \in N(u) : sum(q_j)
            cost_2 = self.cluster_probs[D[u,:]].sum(axis=0)
            old_mean_fields = self.mean_fields.copy()
            self.mean_fields[u,:] = cost_1 - self.lambda_param * cost_2
            # Update cluster probabilities 
            self.cluster_probs[u,:] = self._calculate_cluster_probs(self.mean_fields,self.temperature)[u,:]
            
            # Update cluster centers
            p_yx = np.repeat(self.cluster_probs,n_features).reshape(n_x,n_features*self.n_cluster)
            self.cluster_centers = (np.tile(X,self.n_cluster)*p_yx).sum(axis=0).reshape(-1,n_features) 
            self.cluster_centers /= self.cluster_probs.sum(axis=0).reshape(self.n_cluster,-1)
            
        # Check convergence
        em_delta = np.linalg.norm(self.mean_fields-old_mean_fields)
        
        return em_delta
    
    

    def _calculate_cluster_probs(self, mean_fields, temperature):
        """
        Predict assignment probability vectors for each sample in X given
        the mean fields

        Args:
            mean_fields (np.ndarray): Matrix containing all the mean_fields (n_samples, n_clusters)
            temperature (np.float): Temperature at which probabilities are
                calculated

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (n_samples, n_clusters)
        """
        cluster_probs_not_norm = np.exp(-1/self.temperature * mean_fields)
        Z = np.tile(np.sum(cluster_probs_not_norm,axis=1).reshape(-1,1),self.n_cluster) 
        return cluster_probs_not_norm/Z
    
    def get_neighborhood(self, X):
        """
        Calculate the neighborhood for each object in the feature matrix 
        based on the given metric

        Args:
            X (np.ndarray): Matrix X (n_x, n_features)
        Returns:
            N (np.ndarray): Neighbors (n_x, n_x)
        """

        dist = DistanceMetric.get_metric(self.metric)
        N =  dist.pairwise(X)
        N[N <= self.threshold] = 1
        N[N > self.threshold] = 0
        np.fill_diagonal(N,0)
        N = np.asarray(N, dtype=bool)
        return N
    

    def predict(self):
        """Predict assignment probability vectors for each sample in X.

        Args:

        Returns:
            assignments (np.ndarray): Assignment probability vectors
                (n_x)
        """
        return self.cluster_probs.argmax(axis=1) 


