from sklearn.neighbors import KernelDensity
import numpy as np
import pickle
import codecs

def sample_kde():
    """
    Generates a random KDMA Measurement based on a 
    normally distributed random sample

    The normal distribution is centered on `norm_loc` with a
    a scale of `norm_scale`
    """
    #X = np.array(X) # convert to numpy (if not already)
    N = 100
    X = np.random.normal(0, 1, int(0.3 * N))

    kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(X[:, np.newaxis])

    return kde

def kde_to_base64(kde: KernelDensity) -> str:
    return codecs.encode(pickle.dumps(kde), "base64").decode()

def kde_from_base64(base64_str: str) -> KernelDensity:
    return pickle.loads(codecs.decode(base64_str.encode(), "base64"))

