import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.integrate import trapezoid
import math

def _normalize(x, y):
    """
    Normalize probability distribution y such that its integral over domain x is 1.

    Parameters
    ----------
    x: ndarray
        domain over which discrete probability distribution y is defined.
    
    y: ndarray
        probability distribution at each point in x. Y is proportional to the
        probability density of the distribution at x.
    
    Returns
    --------
    pdf: ndarray
        array with same shape as y that gives normalized probability density function
        values at each point x.

    """
    # area under curve
    auc = trapezoid(y, x)

    # scale y by auc so that new area under curve is 1 --> probability density
    pdf = y / auc

    return pdf


def _kde_to_pdf(kde, x, normalize=True):
    """
    Evaluate kde over domain x and optionally normalize results into pdf.

    Parameters
    ----------
    kde: sklearn KDE model
        model used to generate distribution.

    x: ndarray
        points to evaulate kde at to generate probability function.
    
    
    Returns
    ---------
    pf: ndarray
        array containing probability function evaluated at each element in x.

    """
    pf = np.exp(kde.score_samples(x[:,np.newaxis]))

    if normalize: 
        pf = _normalize(x, pf)

    return pf


def hellinger_similarity(kde1, kde2, samples: int):
    """
    Similarity score derived from the Hellinger distance.

    The Hellinger similarity :math:`H(P,Q)`  between probability density functions
    :math:`P(x)` and :math:`Q(x)` is given by:
    
    .. math::
        H(P,Q) = 1 - D(P,Q)
    
    Where :math:`D(P,Q)` is the
    `hellinger distance <https://en.wikipedia.org/wiki/Hellinger_distance>`_ between
    the distributions.

    The similarity score is bounded between 0 (:math:`P` is 0 everywhere where 
    :math:`Q` is nonzero and vice-versa) and ` (:math:`P(x)=Q(x) \\forall x`)
    
    Parameters
    --------------
    kde1, kde2: sklearn KDE models
        KDEs for distributions to compare.
    
    samples: int
        number of evenly-spaced points on the intevral :math:`[0,1]` 

    """
    # Compute the PDFs of the two KDEs at some common evaluation points
    # How likely each data point is according to the KDE model. Quantifies how well each data point fits the estimated probability distribution.
    x = np.linspace(0, 1, samples)
    pdf_kde1 = _kde_to_pdf(kde1, x)
    pdf_kde2 = _kde_to_pdf(kde2, x)
    

    squared_diff = (np.sqrt(pdf_kde1)-np.sqrt(pdf_kde2))**2
    area = trapezoid(squared_diff, x)
    d_hellinger = np.sqrt(area/2)
    
    return 1 - d_hellinger


def kl_similarity(kde1, kde2, samples: int):
    # Compute the PDFs of the two KDEs at some common evaluation points
    # How likely each data point is according to the KDE model. Quantifies how well each data point fits the estimated probability distribution.
    x = np.linspace(0, 1, samples)
    pdf_kde1 = _kde_to_pdf(kde1, x)
    pdf_kde2 = _kde_to_pdf(kde2, x)
    
    # Compute the Kullback-Leibler Distance using samples
    kl = entropy(pdf_kde1, pdf_kde2)
    # TODO note - KL is not bounded between 0 and 1- inverting may give negative values
    return 1 - kl

# Jensen-Shannon Divergence
def js_similarity(kde1, kde2, samples: int):
    # Compute the PDFs of the two KDEs at some common evaluation points
    # How likely each data point is according to the KDE model. Quantifies how well each data point fits the estimated probability distribution.
    x = np.linspace(0, 1, samples)
    pdf_kde1 = _kde_to_pdf(kde1, x)
    pdf_kde2 = _kde_to_pdf(kde2, x)
    
    # Compute the Jensen-Shannon Distance using samples
    js = jensenshannon(pdf_kde1, pdf_kde2)
    # There is a bug in the jensenshannon code, where sometimes the sqrt function gets passed a very small
    # negative number, which results in a NaN return value.
    # This error case occasionally happens when the two distributions are the same
    if js is None or math.isnan(js):
        js = 0
    # We inverthe value because the spec agreed to with the other ITM performs has
    # 0 = unaligned, 1 = full aligned which is the opposite of what Jensenshannon produces. 
    return 1 - js
