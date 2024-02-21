import numpy as np

from sklearn.metrics.pairwise import rbf_kernel
from scipy import linalg
from scipy.spatial.distance import pdist

def return_sigma(data, perc=90):
    # heuristic to estimate the width of the gaussian kernel
    pairw = pdist(data)
    return np.percentile(pairw,perc)

def nys_idx(X,k, seed):
    rng = np.random.default_rng(seed=seed)
    return rng.choice(X.shape[0], k)     


def nystrom_features(X, idx, kernel_function=rbf_kernel, gamma=1):

    # Nystrom feature map Phi_N = Knm Km^(-1/2)
        
    Knm = kernel_function(X, X[idx],gamma)
    
    Km = Knm[idx]
    
    U, S, V = linalg.svd(Km)
    map = linalg.pinv( U * np.sqrt(S) @ V ).T
    
    return Knm @ map