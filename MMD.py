import numpy as np

from sklearn.gaussian_process.kernels import RBF

from scipy.spatial.distance import pdist


def md_pairwise(data):
    '''
    to estimate the bandwidth
    '''

    return np.median(pdist(data))



def MMD2(x,y,sigma=1):
    '''
    from https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf  (Eq.3)

    '''
    k =  RBF(length_scale=sigma)

    lx = len(x)
    ly = len(y)

    MMD2 = 1/(lx * (lx-1)) * (k(x,x).sum()-k(x,x).trace()) + 1/(ly * (ly-1)) * (k(y,y).sum()-k(y,y).trace()) - 2/(lx * ly) * k(x,y).sum()

    return MMD2


def MMDb(x,y,sigma=1):
    '''
    from https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf  (Eq.5)

    '''
    k =  RBF(length_scale=sigma)

    lx = len(x)
    ly = len(y)

    MMDb = np.sqrt((1/lx^2) * k(x,x).sum() + (1/ly^2) * k(y,y).sum() - 2/(lx * ly) * k(x,y).sum())

    return MMDb
