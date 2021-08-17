import sklearn as skl
from sklearn.utils.validation import check_is_fitted

import pandas as pd
import numpy as np
from treelib import Tree

import matplotlib.pyplot as plt


def read_data_csv(sheet, y_names=None):
    """Parse a column data store into X, y arrays

    Args:
        sheet (str): Path to csv data sheet.
        y_names (list of str): List of column names used as labels.

    Returns:
        X (np.ndarray): Array with feature values from columns that are not
        contained in y_names (n_samples, n_features)
        y (dict of np.ndarray): Dictionary with keys y_names, each key
        contains an array (n_samples, 1) with the label data from the
        corresponding column in sheet.
    """

    data = pd.read_csv(sheet)
    feature_columns = [c for c in data.columns if c not in y_names]
    X = data[feature_columns].values
    y = dict([(y_name, data[[y_name]].values) for y_name in y_names])

    return X, y


class DeterministicAnnealingClustering(skl.base.BaseEstimator,
                                       skl.base.TransformerMixin):
    """Template class for DAC

    Attributes:
        cluster_centers (np.ndarray): Cluster centroids y_i
            (n_clusters, n_features)
        cluster_probs (np.ndarray): Assignment probability vectors
            p(y_i | x) for each sample (n_samples, n_clusters)
        bifurcation_tree (treelib.Tree): Tree object that contains information
            about cluster evolution during annealing.

    Parameters:
        n_clusters (int): Maximum number of clusters returned by DAC.
        random_state (int): Random seed.
    """

    def __init__(self, n_clusters=8, random_state=42,convergence_em=1e-5,alpha=0.99,T_min=10e-1, metric="euclidian"):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.metric = metric
        self.T = None
        self.T_min = T_min
        self.alpha = alpha
        self.convergence_em = convergence_em
        self.cluster_centers = None
        self.cluster_probs = None
        self.cluster_prior = None
        self.open_clusters = 1
        self.n_eff_clusters = list()
        self.temperatures = list()
        self.distortions = list()
        self.bifurcation_tree = Tree()

    def fit(self, samples):
        """Compute DAC for input vectors X

        Preferred implementation of DAC as described in reference [1].

        Args:
            samples (np.ndarray): Input array with shape (samples, n_features)
        """
        #  1) Initialize
        n_features = samples.shape[1]
        n_samples = samples.shape[0]
        samples_norm = samples - np.tile(np.mean(samples,axis=0),(n_samples,1)) # normalize sample matrix
        C = 1/n_samples* (samples_norm.T@samples_norm) # covariance matrix
        u, v = np.linalg.eigh(C) # get eigvals
        T_c = np.max(u) * 2 # critical temperature
        self.cluster_centers = np.mean(samples, axis=0).reshape(-1,n_features) # y_0 = sample_mean
        self.cluster_prior = np.ones(1) # p(y_0) = 1     
        self.T = T_c + T_c*1/10
        self.bifurcation_tree.create_node(tag=0, identifier="0_0", data={"cluster_id":0,"distance":[],"centroid":None})
        
        # 2) Update
        while True: 

            em_delta = self._em_step(samples) # EM algorithm
            
            # 3) Convergence test for EM
            while em_delta >= self.convergence_em: 
                em_delta = self._em_step(samples)

            # 4) Check Temperature
            if self.T <= self.T_min:
                break # no last iteration at T=0 because of numerical instability

            # 5) Cooling Step + Statistics
            self._update_tree() 
            self.temperatures.append(self.T)
            self.n_eff_clusters.append(self.open_clusters)
            D = np.square(self.get_distance(samples,self.cluster_centers))
            self.distortions.append(1/n_samples*np.sum(self.cluster_probs*D))
            self.T *= self.alpha

            # 6) Check critical temperature 
            for i in range(self.open_clusters):                
                if self.open_clusters < self.n_clusters:
                    samples_norm = samples- np.tile(self.cluster_centers[i],(n_samples,1))
                    C =  1/n_samples * 1/self.cluster_prior[i] * ((samples_norm*self.cluster_probs[:,i].reshape(-1,1)).T @ (samples_norm)) # Eq(18) C|y_i
                    u, v = np.linalg.eigh(C) 
                    if self.T <= 2*np.max(u):
                        delta = np.random.normal(0,size=n_features)
                        perturbed_center  = self.cluster_centers[i] + 1e-2*delta
                        self.cluster_centers = np.append(self.cluster_centers,perturbed_center.reshape(1,n_features),axis=0)
                        self.open_clusters += 1
                        self.cluster_prior = np.append(self.cluster_prior,self.cluster_prior[i].reshape(1)/2,axis=0)
                        self.cluster_prior[i] /= 2
                        self._split__tree(i)
                        break # don't split 2 times at the same T


    def _sign(self,x):
        if x >= 0:
            return 1
        return -1 


    def _flatten(self,list_of_lists):
        """Utility function to flatten lists
        Args:
            list_of_lists (list): nested list
        """
        if len(list_of_lists) == 0:
            return list_of_lists
        if isinstance(list_of_lists[0], list):
            return self._flatten(list_of_lists[0]) + self._flatten(list_of_lists[1:])
        return list_of_lists[:1] + self._flatten(list_of_lists[1:])

    def _split__tree(self,idx):
        """Utility function to split the bifurcation tree 
        when a new cluster is created
        Args:
            idx (int): cluster id 
        """
        n_clusters = len(self.bifurcation_tree.leaves())
        c_id = [x.identifier for x in self.bifurcation_tree.leaves() if x.tag == idx][0]
        self.bifurcation_tree.get_node(c_id).data['centroid'] = self.cluster_centers[idx]
        t_depth = self.bifurcation_tree.depth() + 1
        c_id_distances = self.bifurcation_tree.get_node(c_id).data['distance']
        self.bifurcation_tree.create_node(tag=idx, identifier=str(idx)+'_'+str(t_depth), parent=c_id, data={"cluster_id":idx,"distance":[c_id_distances],"dir":1*self._sign(c_id_distances[-1]),"centroid":None})
        self.bifurcation_tree.create_node(tag=n_clusters, identifier=str(n_clusters)+'_'+str(t_depth), parent=c_id, data={"cluster_id":n_clusters,"distance":[c_id_distances[-1]],"dir":-1*self._sign(c_id_distances[-1]), "centroid":None})


    def _update_tree(self):
        leaves = self.bifurcation_tree.leaves()
        for node in leaves:
            if not(node.is_root()):
                p = self.bifurcation_tree.parent(node.identifier)
                # traverse the tree until we find the "real" parent
                # which is the one that first started the cluster_id
                # if this is not the desired behaviour it is enough to comment the while loop
                while p.tag == node.tag and p.tag != 0: 
                    p = self.bifurcation_tree.parent(p.identifier)
                p_centroid = p.data['centroid']
                node_centroid = self.cluster_centers[node.data['cluster_id']]
                sign = node.data['dir']
                node.data['distance'].append(sign*np.linalg.norm(p_centroid-node_centroid))
            else:
                node.data['distance'].append(0)
    
    def _em_step(self,samples):
        n_features = samples.shape[1]
        n_samples = samples.shape[0]
        D = self.get_distance(samples, self.cluster_centers)
        self.cluster_probs = self._calculate_cluster_probs(D,self.T,self.cluster_prior)
        self.cluster_prior = self. _calculate_cluster_prior(self.cluster_probs)
        p_yx = np.repeat(self.cluster_probs,n_features).reshape(n_samples,n_features*self.open_clusters)
        new_cluster_centers = np.mean(np.tile(samples,self.open_clusters)*p_yx,axis=0).reshape(-1,n_features)
        new_cluster_centers *= np.tile( 1/self.cluster_prior.reshape(-1,1),n_features) # prior
        em_delta = np.linalg.norm(self.cluster_centers-new_cluster_centers)
        self.cluster_centers = new_cluster_centers
        return em_delta

    def _calculate_cluster_prior(self,cluster_probs):
            n_samples = cluster_probs.shape[0]
            cluster_prior = 1/n_samples * np.sum(cluster_probs,axis=0)
            return cluster_prior

    def _calculate_cluster_probs(self, dist_mat, temperature, cluster_prior):
        """Predict assignment probability vectors for each sample in X given
            the pairwise distances

        Args:
            dist_mat (np.ndarray): Distances (n_samples, n_centroids)
            temperature (float): Temperature at which probabilities are
                calculated

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (n_samples, open_clusters)
        """

        n_samples = dist_mat.shape[0]
        if self.open_clusters == 1: return np.ones((n_samples,1))

        cluster_probs_not_norm = np.tile(cluster_prior,(n_samples,1)) * np.exp(- np.square(dist_mat)/self.T)
        Z = np.tile(np.sum(cluster_probs_not_norm,axis=1).reshape(-1,1),self.open_clusters) 
        return cluster_probs_not_norm/Z

    def _calculate_euclidean_distance(self, samples, clusters):
        """Calculate the distance matrix between samples and codevectors
        based on the euclidean distance 

        Args:
            samples (np.ndarray): Samples array (n_samples, n_features)
            clusters (np.ndarray): Codebook (n_centroids, n_features)

        Returns:
            D (np.ndarray): Distances (n_samples, n_centroids)
        """

        n_samples = samples.shape[0]
        n_features = samples.shape[1]
        n_centroids = clusters.shape[0]
        C = np.tile(clusters.flatten(),(n_samples,1))
        X = np.tile(samples,(1,n_centroids))
        D = X-C
        return np.linalg.norm(D.reshape(-1,n_features),axis=1).reshape(-1,n_centroids)




    def get_distance(self, samples, clusters):
        """Calculate the distance matrix between samples and codevectors
        based on the given metric

        Args:
            samples (np.ndarray): Samples array (n_samples, n_features)
            clusters (np.ndarray): Codebook (n_centroids, n_features)

        Returns:
            D (np.ndarray): Distances (n_samples, n_centroids)
        """

        distance = {
            "euclidian":  self._calculate_euclidean_distance
        }    
        return distance[self.metric](samples,clusters)

    def predict(self, samples):
        """Predict assignment probability vectors for each sample in X.

        Args:
            samples (np.ndarray): Input array with shape (new_samples, n_features)

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        distance_mat = self.get_distance(samples, self.cluster_centers)
        probs = self._calculate_cluster_probs(distance_mat, self.T_min,self.cluster_prior)
        return probs

    def transform(self, samples):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers

        Args:
            samples (np.ndarray): Input array with shape
                (new_samples, n_features)

        Returns:
            Y (np.ndarray): Cluster-distance vectors (new_samples, n_clusters)
        """
        check_is_fitted(self, ["cluster_centers"])

        distance_mat = self.get_distance(samples, self.cluster_centers)
        return distance_mat

    def plot_bifurcation(self, cut_idx=40):
        """Show the evolution of cluster splitting

        This is a pseudo-code showing how you may be using the tree
        information to make a bifurcation plot. Your implementation may be
        entire different or based on this code.
        """

        check_is_fitted(self, ["bifurcation_tree"])

        clusters = [[] for _ in range(len(np.unique(self.n_eff_clusters)))]
        
        for node in self.bifurcation_tree.leaves():
            c_id = node.data['cluster_id']
            my_dist = node.data['distance']
            clusters[c_id].append(my_dist)

        
        beta = [1 / t for t in self.temperatures]

        for i in range(len(clusters)):
            clusters[i] = self._flatten(clusters[i]) # flatten the distance list
            clusters[i] = np.pad(clusters[i],[len(beta)-len(clusters[i]),0],mode='constant',constant_values=(np.nan,)) # pad with nans

        

        plt.figure(figsize=(10, 5))
        for c_id, s in enumerate(clusters):
            plt.plot(s[:cut_idx], beta[:cut_idx], '-k',
                     alpha=1, c='C%d' % int(c_id),
                     label='Cluster %d' % int(c_id))
        plt.legend()
        plt.xlabel("distance to parent")
        plt.ylabel(r'$1 / T$')
        plt.title('Bifurcation Plot')
        plt.show()

    def plot_phase_diagram(self):
        """Plot the phase diagram

        This is an example of how to make phase diagram plot. The exact
        implementation may vary entirely based on your self.fit()
        implementation. Feel free to make any modifications.
        """
        t_max = np.log(max(self.temperatures))
        d_min = np.log(min(self.distortions))
        y_axis = [np.log(i) - d_min for i in self.distortions]
        x_axis = [t_max - np.log(i) for i in self.temperatures]

        plt.figure(figsize=(12, 9))
        plt.plot(x_axis, y_axis)

        region = {}
        for i, c in list(enumerate(self.n_eff_clusters)):
            if c not in region:
                region[c] = {}
                region[c]['min'] = x_axis[i]
            region[c]['max'] = x_axis[i]
        for c in region:
            if c == 0:
                continue
            plt.text((region[c]['min'] + region[c]['max']) / 2, 0.2,
                     'K={}'.format(c), rotation=90)
            plt.axvspan(region[c]['min'], region[c]['max'], color='C' + str(c),
                        alpha=0.2)
        plt.title('Phases diagram (log)')
        plt.xlabel('Temperature')
        plt.ylabel('Distortion')
        plt.show()
