import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['savefig.dpi'] = 250
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['savefig.bbox'] = 'tight'

import qp

def calculate_moment(p, N, using=None, limits=None, dx=0.01, vb=False):
    """
    Calculates a moment of a qp.PDF object

    Parameters
    ----------
    p: qp.PDF object
        the PDF whose moment will be calculated
    N: int
        order of the moment to be calculated
    limits: tuple of floats
        endpoints of integration interval over which to calculate moments
    dx: float
        resolution of integration grid
    vb: Boolean
        print progress to stdout?

    Returns
    -------
    M: float
        value of the moment
    """
    if limits is None:
        limits = p.limits
    if using is None:
        using = p.first
    # Make a grid from the limits and resolution
    d = int((limits[-1] - limits[0]) / dx)
    grid = np.linspace(limits[0], limits[1], d)
    dx = (limits[-1] - limits[0]) / (d - 1)
    # Evaluate the functions on the grid
    pe = p.evaluate(grid, using=using, vb=vb)[1]
    # pe = normalize_gridded(pe)[1]
    # calculate the moment
    grid_to_N = grid ** N
    M = quick_moment(pe, grid_to_N, dx)
    return M

def quick_moment(p_eval, grid_to_N, dx):
    """
    Calculates a moment of an evaluated PDF

    Parameters
    ----------
    p_eval: numpy.ndarray, float
        the values of a probability distribution
    grid: numpy.ndarray, float
        the grid upon which p_eval was evaluated
    dx: float
        the difference between regular grid points
    N: int
        order of the moment to be calculated

    Returns
    -------
    M: float
        value of the moment
    """
    M = np.dot(grid_to_N, p_eval) * dx
    return M

def calculate_kld(p, q, limits=qp.utils.lims, dx=0.01, vb=False):
    """
    Calculates the Kullback-Leibler Divergence between two qp.PDF objects.

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

    Notes
    -----
    TO DO: change this to calculate_kld
    TO DO: have this take number of points not dx!
    """
    # Make a grid from the limits and resolution
    N = int((limits[-1] - limits[0]) / dx)
    grid = np.linspace(limits[0], limits[1], N)
    dx = (limits[-1] - limits[0]) / (N - 1)
    # Evaluate the functions on the grid and normalize
    pe = p.evaluate(grid, vb=vb, norm=True)
    pn = pe[1]
    qe = q.evaluate(grid, vb=vb, norm=True)
    qn = qe[1]
    # Normalize the evaluations, so that the integrals can be done
    # (very approximately!) by simple summation:
    # pn = pe / np.sum(pe)
    #denominator = max(np.sum(qe), epsilon)
    # qn = qe / np.sum(qe)#denominator
    # Compute the log of the normalized PDFs
    # logquotient = safelog(pn / qn)
    # logp = safelog(pn)
    # logq = safelog(qn)
    # Calculate the KLD from q to p
    Dpq = quick_kld(pn, qn, dx=dx)# np.dot(pn * logquotient, np.ones(len(grid)) * dx)
    if Dpq < 0.:
        print('broken KLD: '+str((Dpq, pn, qn, dx)))
        Dpq = qp.utils.epsilon
    return Dpq

def quick_kld(p_eval, q_eval, dx=0.01):
    """
    Calculates the Kullback-Leibler Divergence between two evaluations of PDFs.

    Parameters
    ----------
    p_eval: numpy.ndarray, float
        evaluations of probability distribution whose distance _from_ `q` will be calculated
    q_eval: numpy.ndarray, float
        evaluations of probability distribution whose distance _to_ `p` will be calculated.
    dx: float
        resolution of integration grid

    Returns
    -------
    Dpq: float
        the value of the Kullback-Leibler Divergence from `q` to `p`

    Notes
    -----
    TO DO: change this to quick_kld
    """
    logquotient = qp.utils.safelog(p_eval) - qp.utils.safelog(q_eval)
    # logp = safelog(pn)
    # logq = safelog(qn)
    # Calculate the KLD from q to p
    Dpq = dx * np.sum(p_eval * logquotient)
    return Dpq

def calculate_rmse(p, q, limits=qp.utils.lims, dx=0.01, vb=False):
    """
    Calculates the Root Mean Square Error between two qp.PDF objects.

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
    vb: boolean
        report on progress to stdout?

    Returns
    -------
    rms: float
        the value of the RMS error between `q` and `p`

    Notes
    -----
    TO DO: change dx to N
    TO DO: change PDF objects to parametrization objects
    """
    # Make a grid from the limits and resolution
    N = int((limits[-1] - limits[0]) / dx)
    grid = np.linspace(limits[0], limits[1], N)
    dx = (limits[-1] - limits[0]) / (N - 1)
    # Evaluate the functions on the grid
    pe = p.evaluate(grid, vb=vb)[1]
    qe = q.evaluate(grid, vb=vb)[1]
    # Calculate the RMS between p and q
    rms = quick_rmse(pe, qe, N)# np.sqrt(dx * np.sum((pe - qe) ** 2))
    return rms

def quick_rmse(p_eval, q_eval, N):
    """
    Calculates the Root Mean Square Error between two evaluations of PDFs.

    Parameters
    ----------
    p_eval: numpy.ndarray, float
        evaluation of probability distribution function whose distance between
        its truth and the approximation of `q` will be calculated.
    q_eval: numpy.ndarray, float
        evaluation of probability distribution function whose distance between
        its approximation and the truth of `p` will be calculated.
    N: int
        number of points at which PDFs were evaluated

    Returns
    -------
    rms: float
        the value of the RMS error between `q` and `p`
    """
    # Calculate the RMS between p and q
    rms = np.sqrt(np.sum((p_eval - q_eval) ** 2) / N)
    return rms

def plot(PDF, limits=None, res=100, names=None, loc='plot.pdf', cmap=None,
        style=False, vb=False):
    """
    Plots representations of a PDF

    Parameters
    ----------
    PDF: qp.PDF object
        the PDF object whose representations will be plotted
    limits: tuple, float, optional
        limits over which plot should be made
    res: int, optional
        number of points at which to evaluate representations
    names: list, string, optional
        names of the parametrizations to include in plot
    loc: string, optional
        filepath/name for saving the plot, optional
    cmap: string, optional
        name of colormap, must be an option from matplotlib colormaps
    style: boolean, optional
        plot formats with different linestyles?
    vb: boolean, optional
        report on progress to stdout?

    Notes
    -----
    What this method plots depends on what information about the PDF
    is stored in it: the more parametrizations the PDF has,
    the more exciting the plot!
    """
    if limits is None:
        limits = PDF.limits
    x = np.linspace(limits[0], limits[-1], res)

    if names is None:
        names = PDF.parametrizations.keys()
    n_p = len(names)
    p_indices = range(n_p)

    if cmap is None:
        cmap = plt.get_cmap('nipy')
    elif type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    if not style:
        style = ['-'] * range(n_p)
    else:
        style = [(0, (i+1, 2*(i+1))) for i in p_indices]

    p_colors = np.linspace(0., 1., n_p)
    colors, styles = {}, {}
    for i in range(n_p):
        colors[names[i]] = cmap(p_colors[i])
        styles[names[i]] = styles[i]

    # colors = {}
    # colors['truth'] = 'k'
    # colors['mix_mod'] = 'k'
    # colors['gridded'] = 'k'
    # colors['quantiles'] = 'blueviolet'
    # colors['histogram'] = 'darkorange'
    # colors['samples'] = 'forestgreen'
    #
    # styles = {}
    # styles['truth'] = '-'
    # styles['mix_mod'] = ':'
    # styles['gridded'] = '--'
    # styles['quantiles'] = '--'#(0,(5,10))
    # styles['histogram'] = ':'#(0,(3,6))
    # styles['samples'] = '-.'#(0,(1,2))

    x = np.linspace(limits[0], limits[-1], 100)
    for name in names:
        y = PDF.parametrizations[name].convert('pointeval', x).parameters
        plt.plot(x, y, color=colors[name], linestyle=styles[name], lw=2.0, alpha=0.5, label=name)
        if vb: print 'Plotted ' + name + '.'


    # if PDF.mixmod is not None:
    #     [min_x, max_x] = [PDF.mixmod.ppf(np.array([0.001])), PDF.mixmod.ppf(np.array([0.999]))]
    #     extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
    #     [min_x, max_x] = extrema
    #     x = np.linspace(min_x, max_x, res)
    #     y = PDF.mixmod.pdf(x)
    #     plt.plot(x, y, color=colors['truth'], linestyle=styles['truth'], lw=5.0, alpha=0.25, label='True PDF')
    #
    # if PDF.mix_mod is not None:
    #     [min_x, max_x] = [PDF.mix_mod.ppf(np.array([0.001])), PDF.mix_mod.ppf(np.array([0.999]))]
    #     extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
    #     [min_x, max_x] = extrema
    #     x = np.linspace(min_x, max_x, 100)
    #     y = PDF.mix_mod.pdf(x)
    #     plt.plot(x, y, color=colors['mix_mod'], linestyle=styles['mix_mod'], lw=2.0, alpha=1.0, label='Mixture Model PDF')
    #
    # if PDF.quantiles is not None:
    #     # (z, p) = PDF.evaluate(PDF.quantiles[1], using='quantiles', vb=vb)
    #     # print('first: '+str((z,p)))
    #     (x, y) = qp.utils.normalize_quantiles(PDF.quantiles)
    #     print('second: '+str((x, y)))
    #     [min_x, max_x] = [min(x), max(x)]
    #     extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
    #     [min_x, max_x] = extrema
    #     x = np.linspace(min_x, max_x, 100)
    #     print('third: '+str(x))
    #     (grid, qinterpolated) = PDF.approximate(x, vb=vb, using='quantiles')
    #     plt.scatter(PDF.quantiles[1], np.zeros(np.shape(PDF.quantiles[1])), color=colors['quantiles'], marker='|', s=100, label='Quantiles', alpha=0.75)
    #     # plt.vlines(z, np.zeros(len(PDF.quantiles[1])), p, color=colors['quantiles'], linestyle=styles['quantiles'], lw=1.0, alpha=1.0, label='Quantiles')
    #     plt.plot(grid, qinterpolated, color=colors['quantiles'], lw=2.0, alpha=1.0, linestyle=styles['quantiles'], label='Quantile Interpolated PDF')
    #
    # if PDF.histogram is not None:
    #     [min_x, max_x] = [min(PDF.histogram[0]), max(PDF.histogram[0])]
    #     extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
    #     [min_x, max_x] = extrema
    #     x = np.linspace(min_x, max_x, 100)
    #     # plt.vlines(PDF.histogram[0], PDF.histogram[0][:-1],
    #     #            PDF.histogram[0][1:], color=colors['histogram'], linestyle=styles['histogram'], lw=1.0, alpha=1., label='histogram bin ends')
    #     plt.scatter(PDF.histogram[0], np.zeros(np.shape(PDF.histogram[0])), color=colors['histogram'], marker='|', s=100, label='Histogram Bin Ends', alpha=0.75)
    #     (grid, hinterpolated) = PDF.approximate(x, vb=vb,
    #                                              using='histogram')
    #     plt.plot(grid, hinterpolated, color=colors['histogram'], lw=2.0, alpha=1.0,
    #              linestyle=styles['histogram'],
    #              label='Histogram Interpolated PDF')
    #     extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
    #
    # if PDF.gridded is not None:
    #     [min_x, max_x] = [min(PDF.gridded[0]), max(PDF.gridded[0])]
    #     extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
    #     [min_x, max_x] = extrema
    #     (x, y) = PDF.gridded
    #     plt.plot(x, y, color=colors['gridded'], lw=1.0, alpha=0.5,
    #              linestyle=styles['gridded'], label='Gridded PDF')
    #     if vb:
    #         print 'Plotted gridded.'
    #
    # if PDF.samples is not None:
    #     [min_x, max_x] = [min(PDF.samples), max(PDF.samples)]
    #     extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
    #     [min_x, max_x] = extrema
    #     x = np.linspace(min_x, max_x, 100)
    #     plt.scatter(PDF.samples, np.zeros(np.shape(PDF.samples)), color=colors['samples'], marker='|', s=100, label='Samples', alpha=0.75)
    #     (grid, sinterpolated) = PDF.approximate(x, vb=vb,
    #                                              using='samples')
    #     plt.plot(grid, sinterpolated, color=colors['samples'], lw=2.0,
    #                 alpha=1.0, linestyle=styles['samples'],
    #                 label='Samples Interpolated PDF')
    #     if vb:
    #         print('Plotted samples')

    plt.xlim(extrema[0], extrema[-1])
    plt.legend(fontsize='large')
    plt.xlabel(r'$z$', fontsize=16)
    plt.ylabel(r'$p(z)$', fontsize=16)
    plt.tight_layout()
    plt.savefig(loc, dpi=250)

    return
