import sklearn as skl
import skimage
import numpy as np
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import linear_sum_assignment
from itertools import combinations
from scipy.special import kl_div
import matplotlib.pyplot as plt

class HistogramClustering(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """Template class for HistogramClustering (HC)
    
    Attributes:
        centroids (np.ndarray): Array of centroid distributions p(y|c) with shape (n_clusters, n_bins).
        
    Parameters:
        n_clusters (int): Number of clusters (textures).
        n_bins (int): Number of bins used to discretize the range of pixel values found in input image X.
        window_size (int): Size of the window used to compute the local histograms for each pixel.
                           Should be an odd number larger or equal to 3.
        random_state (int): Random seed.
        estimation (str): Whether to use Maximum a Posteriori ("MAP") or
                          Deterministic Annealing ("DA") estimation.
    """
    
    def __init__(self, n_clusters=10, n_bins=256, window_size=11,em_convergence=1e-2, random_state=11, estimation="MAP", T_min=1e-2, T_0=10, alpha=0.5,x_test=None):
        self.n_clusters = n_clusters
        self.n_bins =n_bins
        self.window_size = window_size
        self.random_state = random_state
        self.estimation = estimation
        self.cluster_probs = None # c(x)
        self.p_yx = None # p(y|x)
        self.p_yc = None # p(y|c)
        self.em_convergence = em_convergence
        self.T_min = T_min
        self.T = T_0 
        self.alpha = alpha
        self.x_test = x_test

    def fit(self, X):
        """Compute HC for input image X
        
        Compute centroids.        
        
        Args:
            X (np.ndarray): Input array with shape (height, width)
        
        Returns:
            self
        """
        
        np.random.seed(self.random_state)
        
        # compute p(y|x)
        self.p_yx = self._compute_histograms(X)
               
        if self.estimation == "MAP":
            # Maximum a Posteriori
            self._fit_MAP(X)

        elif self.estimation == "DA":
            # Deterministic Annealing
            self._fit_DA(X)
        
        return self
    
    def _compute_histograms(self,X):
        """
        Compute histograms for input image X

        Args: 
        X (np.ndarray): Input array with shape (height, width)

        Returns:
            p_yx (np.ndarray): Histograms array with shape (height, width, n_bins)
        """
        height = X.shape[0]
        width = X.shape[1]
        offset = (self.window_size-1)//2
       
        # compute flat image windows (height,width,window_size^2)  
        windows = skimage.util.view_as_windows(np.pad(X,(self.window_size-1,self.window_size-1),\
                                               mode='constant',constant_values=-1)\
                                              , self.window_size)[offset:height+offset, offset:width+offset]\
                                            .reshape(height,width,-1)
        
        # compute p(y|x) 
        p_yx = np.zeros((height,width,self.n_bins))
        for i in range(height):
            for j in range(width):
                p_yx[i][j] = np.histogram(windows[i][j][windows[i][j] != -1], \
                                               bins=np.linspace(0,1,self.n_bins+1))[0]
                p_yx[i][j] += 1e-2 # laplace smoothing => no 0s in PMFs
                p_yx[i][j] /= np.sum(p_yx[i][j])
        
        return p_yx
    
    def _compute_assigments(self,p_yx):
        """
        Compute cluster assignments.        
        
        Args:
            p_yx (np.ndarray): Histograms array with shape (height, width, n_bins)
        
        Returns:
            cluster_probs (np.ndarray): Cluster assignments (height, width)
        """
        height = p_yx.shape[0]
        width = p_yx.shape[1]
        
        # Maximum a Posteriori
        if self.estimation == "MAP":
            cluster_probs = (np.tile(p_yx,self.n_clusters)* \
                            (-np.log(1e-10+self.p_yc.reshape(-1)))) \
                            .reshape(height,width,self.n_clusters,-1) \
                            .sum(axis=3).argmin(axis=2)

        # Deterministic Annealing 
        else:
            p_yx = np.moveaxis(np.tile(p_yx,(self.n_clusters,1,1,1)),0,2)
            kl = self._compute_KL(p_yx)
            cluster_probs = np.exp(-(kl)/self.T)
            cluster_probs /= cluster_probs.sum(axis=2).reshape(height,width,1)
            cluster_probs = cluster_probs.argmax(axis=2)

        return cluster_probs
           
    def _fit_MAP(self, X):
        """
        Compute HC for input image X with MAP estimation.
        EM algorithm for optimization. 
        
        Args:
            X (np.ndarray): Input array with shape (height, width)
        
        Returns:
            self
        """

        height = X.shape[0]
        width = X.shape[1]
        delta_em = self.em_convergence+1
       
        # initialize centroids p(y|c)
        #self.p_yc = np.random.dirichlet(np.ones(self.n_bins)/self.n_bins, size=self.n_clusters) + 1/self.n_bins # Laplace Smoothing
        self.p_yc = np.random.rand(self.n_clusters, self.n_bins)
        self.p_yc /= self.p_yc.sum(axis=1).reshape(-1,1)

        
        while delta_em >= self.em_convergence:
            
            # compute assignments c(x) \forall x 
            self.cluster_probs = self._compute_assigments(self.p_yx)
            
            old_p_yc = np.copy(self.p_yc)
            one_hot_cluster_probs =  np.eye(self.cluster_probs.max()+1)[self.cluster_probs]
            
            # compute centroids p(y|c)
            self.p_yc = (np.tile(self.p_yx,self.n_clusters).reshape(height,width,self.n_clusters,-1)*\
            one_hot_cluster_probs.reshape(height,width,self.n_clusters,-1))\
            .sum(axis=(0,1))/ one_hot_cluster_probs.sum(axis=(0,1)).reshape(self.n_clusters,-1)
            # check convergence 
            delta_em = np.linalg.norm(old_p_yc-self.p_yc)
    
    def _perturb_centroids(self):
        """
        Perturb cluster centroids (p_yc) when they are too close 
        during annealing. 
        Args:
            self
        Returns:
            self 
        """

        clusters_idx = [i for i in range(self.n_clusters)]
        epsilon = 1e-1 # centroid perturbation
        
        for i,j in combinations(clusters_idx, 2):  # pairs of 2
                if np.linalg.norm(self.p_yc[i] - self.p_yc[j]) <= epsilon:
                    self.p_yc[j] += epsilon * abs(np.random.normal(size=self.n_bins))
                    self.p_yc[j] /= self.p_yc[j].sum()
        return self

    def _fit_DA(self, X):
        """
        Compute HC for input image X with DA estimation.
        
        Args:
            X (np.ndarray): Input array with shape (height, width)
        
        Returns:
            self 
        """
        height = self.p_yx.shape[0]
        width = self.p_yx.shape[1]
        self.p_yc = np.tile(self.p_yx.mean(axis=(0,1)),self.n_clusters).reshape(self.n_clusters,-1) 
        p_yx_test  = self._compute_histograms(self.x_test)

        # Annealing
        while self.T >= self.T_min:

            # Perturb centroids if too close
            self._perturb_centroids()
            
            # While Expectation-Maximization not converged
            em_delta = 1
            while em_delta >= self.em_convergence:
                old_p_yc = np.copy(self.p_yc)
                # Estimate cluster assignments
                p_yx = np.moveaxis(np.tile(self.p_yx,(self.n_clusters,1,1,1)),0,2)
                kl = self._compute_KL(p_yx)
                self.cluster_probs = np.exp(-(kl)/self.T)
                self.cluster_probs /= self.cluster_probs.sum(axis=2).reshape(height,width,1)
                
                # Maximize Entropy
                self.p_yc = (self.cluster_probs.reshape(height,width,self.n_clusters,1) * p_yx).sum(axis=(0,1))
                self.p_yc /= self.cluster_probs.sum(axis=(0,1)).reshape(self.n_clusters,1) 
                 
                # Check delta
                em_delta = np.linalg.norm(old_p_yc-self.p_yc)

            # Plot test prediction at current T
            labels = self.predict(self.x_test, p_yx=p_yx_test)
            plt.imshow(labels, cmap="tab20")
            plt.title("Deterministic Annealing at T: {T:.3f}".format(T = self.T))
            plt.show()

            self.T *= self.alpha
            
        
        return self
        
    def _compute_KL(self, p_yx):
        """
        Compute KL divergence between two histograms.
        
        Args:
            p_yx (np.ndarray): Feature histogram with shape (height, width, n_cluster, n_bins)
        
        Returns:
            kl (np.ndarray): KL Divergence Matrix (height, width, n_clusters)
        """
        height = p_yx.shape[0]
        width = p_yx.shape[1]
        
        p_yc = np.tile(self.p_yc,(height,width,1,1))
        kl = (p_yx * np.log(p_yx/p_yc)).sum(axis=3)

        return kl

    def _compute_cluster_mapping(self, c1, c2):
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
        C = np.zeros((self.n_clusters,self.n_clusters))
        for i in range(self.n_clusters):
            c1_i = np.nonzero(c1==i)[0]
            for j in range(self.n_clusters):
                c2_j = np.nonzero(c2==j)[0]
                C[i][j] = c1_i.size + c2_j.size - 2*np.intersect1d(c1_i,c2_j).size
        
        row_ind, col_ind = linear_sum_assignment(C)
        return col_ind

    def transform_prediction(self, pred, gt):
        """
        Transform the predicted cluster assignments in order to match 
        with the ground truth assignments.

        Args:
            pred (np.ndarray): Predicted cluster assignments with shape (height, width)
            gt (np.ndarray): Ground truth cluster assignments with shape (height, width)
        Returns:
            transformed (np.ndarray): Predicted cluster assignments mapped to ground truth assignments (height, width) 
        """
        height = pred.shape[0]
        width = pred.shape[1]
        flat_pred = pred.flatten()
        flat_gt = gt.flatten()
        c_map = self._compute_cluster_mapping(flat_pred, flat_gt)
        transformed = np.asarray([c_map[int(i)] for i in flat_pred]) # pred -> gt
        return transformed.reshape(height, width)

    def predict(self, X, p_yx=None):

        """Predict cluster assignments for each pixel in image X.
        
        Args:
            X (np.ndarray): Input array with shape (height, width)
            
        Returns:
            C (np.ndarray): Assignment map (height, width) 
        """
        check_is_fitted(self, ["p_yc"])
        
        if p_yx is None:
            p_yx = self._compute_histograms(X)

        return self._compute_assigments(p_yx)
    
    def generate(self, C):
        """Generate a sample image X from a texture label map C.
        
        The entries of C are integers from the set {1,...,n_clusters}. They represent the texture labels
        of each pixel. Given the texture labels, a sample image X is generated by sampling
        the value of each pixel from the fitted p(y|c).
        
        Args:
            C (np.ndarray): Input array with shape (height, width)
            
        Returns:
            X (np.ndarray): Sample image (height, width)
        """

        check_is_fitted(self, ["p_yc"])
        height = C.shape[0]
        width = C.shape[1]
        X = np.zeros((height,width))
        bin_space = np.linspace(0,1,self.n_bins+1)

        for i in range(height):
            for j in range(width):
                c = int(C[i][j])
                p = self.p_yc[c]
                z = np.random.choice(range(self.n_bins),p=p) # z ~ Cat(p_1,...,p_k) choose a bin
                x = np.random.uniform(bin_space[z],bin_space[z+1]) # x ~ U(a_z,b_z) over the chosen bin
                X[i][j] = x
        return X