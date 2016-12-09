import numpy as np
import sys

# tools for sampling an arbitrary discrete distribution, used in data generation
def cdf(weights):
    """
    Creates a normalized CDF from an arbitrary discrete distribution

    Parameters
    ----------
    weights: ndarray, float
        array of input weights for classes

    Returns
    -------
    result: ndarray, float
        discrete CDF
    """
    tot = sum(weights)
    result = []
    cumsum = 0.
    for w in weights:
        cumsum += w
        result.append(cumsum/tot)
    return np.array(result)

def choice(pop, weights):
    """
    choice takes a population and assigns each element a value from 0 to len(weights) based on CDF of weights
    """
    assert len(pop) == len(weights)
    cdf_vals = cdf(weights)
    x = np.random.random()
    index = bisect.bisect(cdf_vals,x)
    return pop[index]

def safelog(arr, threshold=sys.float_info.epsilon):
    """
    Takes the natural logarithm of an array that might contain zeroes.

    Parameters
    ----------
    arr: ndarray
        values to be logged
    threshold: float
        small, positive value to replace zeros and negative numbers

    Returns
    -------
    logged: ndarray
        logarithms, with approximation in place of zeros and negative numbers
    """
    shape = np.shape(arr)
    flat = arr.flatten()
    logged = np.log(np.array([max(a,threshold) for a in flat])).reshape(shape)
    return logged

def calculate_kl_divergence(p, q, limits=(-10.0,10.0), dx=0.01, vb=True):
    """
    Calculates the Kullback-Leibler Divergence between two PDFs.

    Parameters
    ----------
    p: PDF object
        probability distribution whose distance _from_ `q` will be calculated.
    q: PDF object
        probability distribution whose distance _to_ `p` will be calculated.
    limits: tuple of floats
        endpoints of integration interval in which to calculate KLD
    dx: float
        resolution of integration grid
    vb: boolean
        report on progress to stdout?

    Returns
    -------
    Dpq: float
        the value of the Kullback-Leibler Divergence from `q` to `p`
    """
    # Make a grid from the limits and resolution
    grid = np.linspace(limits[0], limits[1], int((limits[1]-limits[0])/dx))
    # Evaluate the functions on the grid
    pe = p.evaluate(grid, vb=vb)
    qe = q.evaluate(grid, vb=vb)
    # Normalize the evaluations, so that the integrals can be done
    # (very approximately!) by simple summation:
    pn = pe/np.sum(pe)
    qn = qe/np.sum(qe)
    # Compute the log of the normalized PDFs
    logp = safelog(pn)
    logq = safelog(qn)
    # Calculate the KLD from q to p
    Dpq = np.sum(pn*(logp-logq))
    return Dpq

def calculate_rms(p, q, limits=(-10.,10.), dx=0.01):
    """
    Calculates the Root Mean Square Error between two PDFs.

    Parameters
    ----------
    p: PDF object
        probability distribution function whose distance between its truth and the approximation of `q` will be calculated.
    q: PDF object
        probability distribution function whose distance between its approximation and the truth of `p` will be calculated.
    limits: tuple of floats
        endpoints of integration interval in which to calculate RMS
    dx: float
        resolution of integration grid

    Returns
    -------
    rms: float
        the value of the RMS error between `q` and `p`
    """
    # Make a grid from the limits and resolution
    npoints = int((limits[1]-limits[0])/dx)
    grid = np.linspace(limits[0], limits[1], npoints)
    # Evaluate the functions on the grid
    pe = p.evaluate(grid)
    qe = q.evaluate(grid)
    # Calculate the RMS between p and q
    rms = np.sqrt(np.sum((pe-qe)**2)/npoints)
    return rms
