
#comment out for NERSC
# %load_ext autoreload

#comment out for NERSC
# %autoreload 2

from __future__ import print_function

import hickle
import numpy as np
import random
import cProfile
import pstats
import StringIO
import sys
import os
import timeit
import bisect

import qp
from qp.utils import calculate_kl_divergence as make_kld

# np.random.seed(seed=42)
# random.seed(a=42)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['savefig.dpi'] = 250
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['savefig.bbox'] = 'tight'

#comment out for NERSC
# %matplotlib inline

def setup_dataset(dataset_key, skip_rows, skip_cols):
    start = timeit.default_timer()
    with open(dataset_info[dataset_key]['filename'], 'rb') as data_file:
        lines = (line.split(None) for line in data_file)
        for r in range(skip_rows):
            lines.next()
        pdfs = np.array([[float(line[k]) for k in range(skip_cols, len(line))] for line in lines])
    print('read in data file in '+str(timeit.default_timer()-start))
    return(pdfs)

def make_instantiation(dataset_key, n_gals_use, pdfs, bonus=None):

    start = timeit.default_timer()

    n_gals_tot = len(pdfs)
    full_gal_range = range(n_gals_tot)
    subset = np.random.choice(full_gal_range, n_gals_use, replace=False)#range(n_gals_use)
    pdfs_use = pdfs[subset]

    modality = []
    dpdfs = pdfs_use[:,1:] - pdfs_use[:,:-1]
    iqrs = []
    for i in range(n_gals_use):
        modality.append(len(np.where(np.diff(np.signbit(dpdfs[i])))[0]))
        cdf = np.cumsum(qp.utils.normalize_integral((dataset_info[dataset_key]['z_grid'], pdfs_use[i]), vb=False)[1])
        iqr_lo = dataset_info[dataset_key]['z_grid'][bisect.bisect_left(cdf, 0.25)]
        iqr_hi = dataset_info[dataset_key]['z_grid'][bisect.bisect_left(cdf, 0.75)]
        iqrs.append(iqr_hi - iqr_lo)
    modality = np.array(modality)

    dataset_info[dataset_key]['N_GMM'] = int(np.median(modality))+1
#     print('n_gmm for '+dataset_info[dataset_key]['name']+' = '+str(dataset_info[dataset_key]['N_GMM']))

    # using the same grid for output as the native format, but doesn't need to be so
    dataset_info[dataset_key]['in_z_grid'] = dataset_info[dataset_key]['z_grid']
    dataset_info[dataset_key]['metric_z_grid'] = dataset_info[dataset_key]['z_grid']

    print('preprocessed data in '+str(timeit.default_timer()-start))

    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, 'pzs'+str(n_gals_use)+dataset_key+bonus)
    with open(loc+'.hkl', 'w') as filename:
        info = {}
        info['randos'] = randos
        info['z_grid'] = dataset_info[dataset_key]['in_z_grid']
        info['pdfs'] = pdfs_use
        info['modes'] = modality
        info['iqrs'] = iqrs
        hickle.dump(info, filename)

    return(pdfs_use)

def plot_examples(n_gals_use, dataset_key, bonus=None):

    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, 'pzs'+str(n_gals_use)+dataset_key+bonus)
    with open(loc+'.hkl', 'r') as filename:
        info = hickle.load(filename)
        randos = info['randos']
        z_grid = info['z_grid']
        pdfs = info['pdfs']

    plt.figure()
    for i in range(n_plot):
        data = (z_grid, pdfs[randos[i]])
        data = qp.utils.normalize_integral(qp.utils.normalize_gridded(data))
        pz_max.append(np.max(data))
        plt.plot(data[0], data[1], label=dataset_info[dataset_key]['name']+' \#'+str(randos[i]), color=color_cycle[i])
    plt.xlabel(r'$z$', fontsize=14)
    plt.ylabel(r'$p(z)$', fontsize=14)
    plt.xlim(min(z_grid), max(z_grid))
    plt.ylim(0., max(pz_max))
    plt.title(dataset_info[dataset_key]['name']+' data examples', fontsize=16)
    plt.legend()
    plt.savefig(loc+'.pdf', dpi=250)
    plt.close()

    if 'modes' in info.keys():
        modes = info['modes']
        modes_max.append(np.max(modes))
        plt.figure()
        ax = plt.hist(modes, color='k', alpha=1./n_plot, histtype='stepfilled', bins=range(max(modes_max)+1))
        plt.xlabel('modes')
        plt.ylabel('frequency')
        plt.title(dataset_info[dataset_key]['name']+' data modality distribution (median='+str(dataset_info[dataset_key]['N_GMM'])+')', fontsize=16)
        plt.savefig(loc+'modality.pdf', dpi=250)
        plt.close()

    if 'iqrs' in info.keys():
        iqrs = info['iqrs']
        iqr_min.append(min(iqrs))
        iqr_max.append(max(iqrs))
        plot_bins = np.linspace(min(iqr_min), max(iqr_max), 20)
        plt.figure()
        ax = plt.hist(iqrs, bins=plot_bins, color='k', alpha=1./n_plot, histtype='stepfilled')
        plt.xlabel('IQR')
        plt.ylabel('frequency')
        plt.title(dataset_info[dataset_key]['name']+' data IQR distribution', fontsize=16)
        plt.savefig(loc+'iqrs.pdf', dpi=250)
        plt.close()

def setup_from_grid(dataset_key, in_pdfs, z_grid, N_comps, high_res=1000, bonus=None):

    #read in the data, happens to be gridded
    zlim = (min(z_grid), max(z_grid))
    N_pdfs = len(in_pdfs)

    start = timeit.default_timer()
#     print('making the initial ensemble of '+str(N_pdfs)+' PDFs')
    E0 = qp.Ensemble(N_pdfs, gridded=(z_grid, in_pdfs), limits=dataset_info[dataset_key]['z_lim'], vb=False)
    print('made the initial ensemble of '+str(N_pdfs)+' PDFs in '+str(timeit.default_timer() - start))

    #fit GMMs to gridded pdfs based on samples (faster than fitting to gridded)
    start = timeit.default_timer()
#     print('sampling for the GMM fit')
    samparr = E0.sample(high_res, vb=False)
    print('took '+str(high_res)+' samples in '+str(timeit.default_timer() - start))

    start = timeit.default_timer()
#     print('making a new ensemble from samples')
    Ei = qp.Ensemble(N_pdfs, samples=samparr, limits=dataset_info[dataset_key]['z_lim'], vb=False)
    print('made a new ensemble from samples in '+str(timeit.default_timer() - start))

    start = timeit.default_timer()
#     print('fitting the GMM to samples')
    GMMs = Ei.mix_mod_fit(comps=N_comps, vb=False)
    print('fit the GMM to samples in '+str(timeit.default_timer() - start))

    #set the GMMS as the truth
    start = timeit.default_timer()
#     print('making the final ensemble')
    Ef = qp.Ensemble(N_pdfs, truth=GMMs, limits=dataset_info[dataset_key]['z_lim'], vb=False)
    print('made the final ensemble in '+str(timeit.default_timer() - start))

    path = os.path.join(dataset_key, str(N_pdfs))
    loc = os.path.join(path, 'pzs'+str(n_gals_use)+dataset_key+bonus)
    with open(loc+'.hkl', 'w') as filename:
        info = {}
        info['randos'] = randos
        info['z_grid'] = z_grid
        info['pdfs'] = Ef.evaluate(z_grid, using='truth', norm=True, vb=False)[1]
        hickle.dump(info, filename)

    start = timeit.default_timer()
#     print('calculating '+str(n_moments_use)+' moments of original PDFs')
    in_moments, vals = [], []
    for n in range(n_moments_use):
        in_moments.append(Ef.moment(n, using='truth', limits=zlim,
                                    dx=delta_z, vb=False))
        vals.append(n)
    moments = np.array(in_moments)
    print('calculated '+str(n_moments_use)+' moments of original PDFs in '+str(timeit.default_timer() - start))

    path = os.path.join(dataset_key, str(N_pdfs))
    loc = os.path.join(path, 'pz_moments'+str(n_gals_use)+dataset_key+bonus)
    with open(loc+'.hkl', 'w') as filename:
        info = {}
        info['truth'] = moments
        info['orders'] = vals
        hickle.dump(info, filename)

    return(Ef)

def analyze_individual(E, z_grid, N_floats, dataset_key, N_moments=4, i=None, bonus=None):
    zlim = (min(z_grid), max(z_grid))
    z_range = zlim[-1] - zlim[0]
    delta_z = z_range / len(z_grid)
    path = os.path.join(dataset_key, str(n_gals_use))

    Eq, Eh, Es = E, E, E
    inits = {}
    for f in formats:
        inits[f] = {}
        for ff in formats:
            inits[f][ff] = None

    qstart = timeit.default_timer()
#     print('performing quantization')
    inits['quantiles']['quantiles'] = Eq.quantize(N=N_floats, vb=False)
    print('finished quantization in '+str(timeit.default_timer() - qstart))
    hstart = timeit.default_timer()
#     print('performing histogramization')
    inits['histogram']['histogram'] = Eh.histogramize(N=N_floats, binrange=zlim, vb=False)
    print('finished histogramization in '+str(timeit.default_timer() - hstart))
    sstart = timeit.default_timer()
#     print('performing sampling')
    inits['samples']['samples'] = Es.sample(samps=N_floats, vb=False)
    print('finished sampling in '+str(timeit.default_timer() - sstart))

#     print('making the approximate ensembles')
#     start = timeit.default_timer()
    Eo = {}
    for f in formats:
        fstart = timeit.default_timer()
        Eo[f] = qp.Ensemble(E.n_pdfs, truth=E.truth,
                            quantiles=inits[f]['quantiles'],
                            histogram=inits[f]['histogram'],
                            samples=inits[f]['samples'],
                            limits=dataset_info[dataset_key]['z_lim'])
        fbonus = str(N_floats)+f+str(i)
        loc = os.path.join(path, 'pzs'+str(n_gals_use)+dataset_key+fbonus)
        with open(loc+'.hkl', 'w') as filename:
            info = {}
            info['randos'] = randos
            info['z_grid'] = z_grid
            info['pdfs'] = Eo[f].evaluate(z_grid, using=f, norm=True, vb=False)[1]
            hickle.dump(info, filename)
        print('made '+f+' ensemble in '+str(timeit.default_timer()-fstart))
#     print('made all approximate ensembles in '+str(timeit.default_timer() - start))

#     print('calculating the individual metrics')
    metric_start = timeit.default_timer()
    inloc = os.path.join(path, 'pz_moments'+str(n_gals_use)+dataset_key+bonus)
    with open(inloc+'.hkl', 'r') as infilename:
        pz_moments = hickle.load(infilename)
    klds, metrics, kld_moments = {}, {}, {}

    for key in Eo.keys():
        key_start = timeit.default_timer()
#         print('starting '+key)
        klds[key] = Eo[key].kld(using=key, limits=zlim, dx=delta_z)
        samp_metric = qp.PDF(samples=klds[key])
        gmm_metric = samp_metric.mix_mod_fit(n_components=dataset_info[dataset_key]['N_GMM'],
                                             using='samples', vb=False)
        metrics[key] = qp.PDF(truth=gmm_metric)


        pz_moments[key], kld_moments[key] = [], []
        for n in range(N_moments):
            kld_moments[key].append(qp.utils.calculate_moment(metrics[key], n,
                                                          using='truth',
                                                          limits=zlim,
                                                          dx=delta_z,
                                                          vb=False))
            pz_moments[key].append(Eo[key].moment(n, using=key, limits=zlim,
                                                  dx=delta_z, vb=False))
        print('calculated the '+key+' individual moments, kld moments in '+str(timeit.default_timer() - key_start))
#     print('calculated the individual metrics in '+str(timeit.default_timer() - metric_start))

    loc = os.path.join(path, 'kld_hist'+str(n_gals_use)+dataset_key+str(N_floats)+'_'+str(i))
    with open(loc+'.hkl', 'w') as filename:
        info = {}
        info['z_grid'] = z_grid
        info['N_floats'] = N_floats
        info['pz_klds'] = klds
        hickle.dump(info, filename)

    outloc = os.path.join(path, 'pz_moments'+str(n_gals_use)+dataset_key+str(N_floats)+'_'+str(i))
    with open(outloc+'.hkl', 'w') as outfilename:
        hickle.dump(pz_moments, outfilename)

    return(Eo, klds, kld_moments, pz_moments)

def plot_individual_kld(n_gals_use, dataset_key, N_floats, i):

    path = os.path.join(dataset_key, str(n_gals_use))
    a = 1./len(formats)
    loc = os.path.join(path, 'kld_hist'+str(n_gals_use)+dataset_key+str(N_floats)+'_'+str(i))
    with open(loc+'.hkl', 'r') as filename:
        info = hickle.load(filename)
        z_grid = info['z_grid']
        N_floats = info['N_floats']
        pz_klds = info['pz_klds']

    plt.figure()
    plot_bins = np.linspace(-3., 3., 20)
    for key in pz_klds.keys():
        logdata = qp.utils.safelog(pz_klds[key])
        kld_hist = plt.hist(logdata, color=colors[key], alpha=a, histtype='stepfilled', edgecolor='k',
             label=key, normed=True, bins=plot_bins, linestyle=stepstyles[key], ls=stepstyles[key], lw=2)
        hist_max.append(max(kld_hist[0]))
        dist_min.append(min(logdata))
        dist_max.append(max(logdata))
    plt.legend()
    plt.ylabel('frequency', fontsize=14)
    plt.xlabel(r'$\log[KLD]$', fontsize=14)
#     plt.xlim(min(dist_min), max(dist_max))
#     plt.ylim(0., max(hist_max))
    plt.title(dataset_info[dataset_key]['name']+r' data $p(KLD)$ with $N_{f}='+str(N_floats)+r'$', fontsize=16)
    plt.savefig(loc+'.pdf', dpi=250)
    plt.close()

def plot_individual_moment(n_gals_use, dataset_key, N_floats, i):

    path = os.path.join(dataset_key, str(n_gals_use))
    a = 1./len(formats)
    loc = os.path.join(path, 'pz_moments'+str(n_gals_use)+dataset_key+str(N_floats)+'_'+str(i))
    with open(loc+'.hkl', 'r') as filename:
        moments = hickle.load(filename)
    delta_moments = {}

    plt.figure(figsize=(5, 5 * (n_moments_use-1)))
    for n in range(1, n_moments_use):
        ax = plt.subplot(n_moments_use, 1, n)
        ends = (min(moments['truth'][n]), max(moments['truth'][n]))
        for key in formats:
            ends = (min(ends[0], min(moments[key][n])), max(ends[-1], max(moments[key][n])))
        plot_bins = np.linspace(ends[0], ends[-1], 20)
        ax.hist([-100], color='k', alpha=a, histtype='stepfilled', edgecolor='k', label='truth',
                    linestyle='-', ls='-')
        ax.hist(moments['truth'][n], bins=plot_bins, color='k', alpha=a, histtype='stepfilled', normed=True)
        ax.hist(moments['truth'][n], bins=plot_bins, color='k', histtype='step', normed=True, linestyle='-', alpha=a)
        for key in formats:
            ax.hist([-100], color=colors[key], alpha=a, histtype='stepfilled', edgecolor='k', label=key,
                    linestyle=stepstyles[key], ls=stepstyles[key], lw=2)
            ax.hist(moments[key][n], bins=plot_bins, color=colors[key], alpha=a, histtype='stepfilled', normed=True)
            ax.hist(moments[key][n], bins=plot_bins, color='k', histtype='step', normed=True, linestyle=stepstyles[key], alpha=a, lw=2)
        ax.legend()
        ax.set_ylabel('frequency', fontsize=14)
        ax.set_xlabel(moment_names[n], fontsize=14)
        ax.set_xlim(min(plot_bins), max(plot_bins))
    plt.suptitle(dataset_info[dataset_key]['name']+r' data moments with $N_{f}='+str(N_floats)+r'$', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(loc+'.pdf', dpi=250)
    plt.close()

    plt.figure(figsize=(5, 5 * (n_moments_use-1)))
    for n in range(1, n_moments_use):
        ax = plt.subplot(n_moments_use, 1, n)
        ends = (100., -100.)
        for key in formats:
            delta_moments[key] = moments[key] - moments['truth']
            ends = (min(ends[0], min(delta_moments[key][n])), max(ends[-1], max(delta_moments[key][n])))
        plot_bins = np.linspace(ends[0], ends[-1], 20)
        for key in formats:
            ax.hist([-100], color=colors[key], alpha=a, histtype='stepfilled', edgecolor='k', label=key,
                    linestyle=stepstyles[key], ls=stepstyles[key], lw=2)
            ax.hist(delta_moments[key][n], bins=plot_bins, color=colors[key], alpha=a, histtype='stepfilled', normed=True)
            ax.hist(delta_moments[key][n], bins=plot_bins, color='k', histtype='step', normed=True, linestyle=stepstyles[key], alpha=a, lw=2)
        ax.legend()
        ax.set_ylabel('frequency', fontsize=14)
        ax.set_xlabel(r'$\Delta$ '+moment_names[n], fontsize=14)
        ax.set_xlim(min(plot_bins), max(plot_bins))
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle(dataset_info[dataset_key]['name']+r' data moment differences with $N_{f}='+str(N_floats)+r'$', fontsize=16)
    plt.savefig(loc+'_delta.pdf', dpi=250)
    plt.close()

def analyze_stacked(E0, E, z_grid, n_floats_use, dataset_key, i=None):

    zlim = (min(z_grid), max(z_grid))
    z_range = zlim[-1] - zlim[0]
    delta_z = z_range / len(z_grid)

#     print('stacking the ensembles')
#     stack_start = timeit.default_timer()
    stacked_pdfs, stacks = {}, {}
    for key in formats:
        start = timeit.default_timer()
        stacked_pdfs[key] = qp.PDF(gridded=E[key].stack(z_grid, using=key,
                                                        vb=False)[key])
        stacks[key] = stacked_pdfs[key].evaluate(z_grid, using='gridded', norm=True, vb=False)[1]
        print('stacked '+key+ ' in '+str(timeit.default_timer()-start))
    stack_start = timeit.default_timer()
    stacked_pdfs['truth'] = qp.PDF(gridded=E0.stack(z_grid, using='truth',
                                                    vb=False)['truth'])

    stacks['truth'] = stacked_pdfs['truth'].evaluate(z_grid, using='gridded', norm=True, vb=False)[1]
    print('stacked truth in '+str(timeit.default_timer() - stack_start))

#     print('calculating the metrics')
    metric_start = timeit.default_timer()
    klds, moments = {}, {}
    moments['truth'] = []
    for n in range(n_moments_use):
        moments['truth'].append(qp.utils.calculate_moment(stacked_pdfs['truth'], n,
                                                          limits=zlim,
                                                          dx=delta_z,
                                                          vb=False))
    print('calculated the true moments in '+str(timeit.default_timer() - metric_start))
    for key in formats:
        metric_start = timeit.default_timer()
        klds[key] = qp.utils.calculate_kl_divergence(stacked_pdfs['truth'],
                                                     stacked_pdfs[key],
                                                     limits=zlim, dx=delta_z)
        moments[key] = []
        for n in range(n_moments_use):
            moments[key].append(qp.utils.calculate_moment(stacked_pdfs[key], n,
                                                          limits=zlim,
                                                          dx=delta_z,
                                                          vb=False))
        print('calculated the '+key+' stacked kld and moments in '+str(timeit.default_timer() - metric_start))

    path = os.path.join(dataset_key, str(E0.n_pdfs))
    loc = os.path.join(path, 'nz_comp'+str(n_gals_use)+dataset_key+str(n_floats_use)+'_'+str(i))
    with open(loc+'.hkl', 'w') as filename:
        info = {}
        info['z_grid'] = z_grid
        info['stacks'] = stacks
        info['klds'] = klds
        info['moments'] = moments
        hickle.dump(info, filename)

    return(stacked_pdfs, klds, moments)

def plot_estimators(n_gals_use, dataset_key, n_floats_use, i=None):

    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, 'nz_comp'+str(n_gals_use)+dataset_key+str(n_floats_use)+'_'+str(i))
    with open(loc+'.hkl', 'r') as filename:
        info = hickle.load(filename)
        z_grid = info['z_grid']
        stacks = info['stacks']
        klds = info['klds']

    plt.figure()
    plt.plot(z_grid, stacks['truth'], color='black', lw=3, alpha=0.3, label='truth')
    nz_max.append(max(stacks['truth']))
    for key in formats:
        nz_max.append(max(stacks[key]))
        plt.plot(z_grid, stacks[key], label=key+r' KLD='+str(klds[key]), color=colors[key], linestyle=styles[key])
    plt.xlabel(r'$z$', fontsize=14)
    plt.ylabel(r'$\hat{n}(z)$', fontsize=14)
    plt.xlim(min(z_grid), max(z_grid))
#     plt.ylim(0., max(nz_max))
    plt.legend()
    plt.title(dataset_info[dataset_key]['name']+r' data $\hat{n}(z)$ with $N_{f}='+str(n_floats_use)+r'$', fontsize=16)
    plt.savefig(loc+'.pdf', dpi=250)
    plt.close()

def save_moments(dataset_key, n_gals_use, N_f, stat, stat_name):

    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, stat_name+str(n_gals_use)+dataset_key)

    if os.path.exists(loc+'.hkl'):
        with open(loc+'.hkl', 'r') as stat_file:
        #read in content of list/dict
            stats = hickle.load(stat_file)
    else:
        stats = {}
        stats['N_f'] = []
        for f in stat.keys():
            stats[f] = []
            for m in range(n_moments_use):
                stats[f].append([])

    if N_f not in stats['N_f']:
        stats['N_f'].append(N_f)
        for f in stat.keys():
            for m in range(n_moments_use):
                stats[f][m].append([])

    where_N_f = stats['N_f'].index(N_f)

    for f in stat.keys():
        for m in range(n_moments_use):
            stats[f][m][where_N_f].append(stat[f][m])

    with open(loc+'.hkl', 'w') as stat_file:
        hickle.dump(stats, stat_file)

def plot_pz_metrics(dataset_key, n_gals_use):

    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, 'pz_kld_moments'+str(n_gals_use)+dataset_key)
    with open(loc+'.hkl', 'r') as pz_file:
        pz_stats = hickle.load(pz_file)
#     if len(instantiations) == 10:
#         for f in formats:
#             for n in range(n_moments_use):
#                 if not np.shape(pz_stats[f][n]) == (4, 10):
#                     for s in range(len(pz_stats[f][n])):
#                         pz_stats[f][n][s] = np.array(np.array(pz_stats[f][n][s])[:10]).flatten()

    flat_floats = np.array(pz_stats['N_f']).flatten()
    in_x = np.log(flat_floats)

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    shapes = moment_shapes
    marksize = 10
    a = 1./len(formats)

    fig, ax = plt.subplots()
    fig.subplots_adjust(right=1.)
    ax_n = ax
    for key in formats:
        ax.plot([-1], [0], color=colors[key], label=key, linewidth=1, linestyle=styles[key])
    for n in range(1, n_moments_use):
        ax.scatter([-1], [0], color='k', alpha=1., marker=shapes[n], s=marksize, label=moment_names[n])
        n_factor = 0.1 * (n - 2)
        if n>1:
            ax_n = ax.twinx()
        if n>2:
            ax_n.spines["right"].set_position(("axes", 1. + 0.1 * (n-1)))
            make_patch_spines_invisible(ax_n)
            ax_n.spines["right"].set_visible(True)
        for s in range(len(formats)):
            f = formats[s]
            f_factor = 0.05 * (s - 1)
            data_arr = np.log(np.swapaxes(np.array(pz_stats[f][n]), 0, 1))#go from n_floats*instantiations to instantiations*n_floats
            mean = np.mean(data_arr, axis=0).flatten()
            std = np.std(data_arr, axis=0).flatten()
            y_plus = mean + std
            y_minus = mean - std
            y_cor = np.array([y_minus[:-1], y_plus[:-1], y_plus[1:], y_minus[1:]])
            ax_n.plot(np.exp(in_x+n_factor), mean, marker=shapes[n], markersize=marksize, linestyle=styles[f], alpha=2. * a, color=colors[f])
            ax_n.vlines(np.exp(in_x+n_factor), y_minus, y_plus, linewidth=3., alpha=a, color=colors[f])
            pz_mean_max[n] = max(pz_mean_max[n], np.max(y_plus))
            pz_mean_min[n] = min(pz_mean_min[n], np.min(y_minus))
        ax_n.set_ylabel(r'$\log[\mathrm{'+moment_names[n]+r'}]$', fontsize=14)
        ax_n.set_ylim((pz_mean_min[n]-1., pz_mean_max[n]+1.))
    ax.set_xscale('log')
    ax.set_xticks(flat_floats)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(np.exp(min(in_x)-0.25), np.exp(max(in_x)+0.25))
    ax.set_xlabel('number of parameters', fontsize=14)
    ax.set_title(dataset_info[dataset_key]['name']+r' data $\log[KLD]$ moments', fontsize=16)
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(loc+'_clean.pdf', dpi=250)
    plt.close()

    fig, ax = plt.subplots()
    fig.subplots_adjust(right=1.)
    ax_n = ax
    for key in formats:
        ax_n.plot([-1], [0], color=colors[key], label=key, linestyle=styles[key], linewidth=1)
    for n in range(1, n_moments_use):
        n_factor = 0.1 * (n - 2)
        ax.scatter([-1], [0], color='k', marker=shapes[n], s=marksize, label='moment '+str(n))
        if n>1:
            ax_n = ax.twinx()
        if n>2:
            ax_n.spines["right"].set_position(("axes", 1. + 0.1 * (n-1)))
            make_patch_spines_invisible(ax_n)
            ax_n.spines["right"].set_visible(True)
        for s in range(len(formats)):
            f = formats[s]
            f_factor = 0.05 * (s - 1)
            data_arr = np.log(np.swapaxes(np.array(pz_stats[f][n]), 0, 1))#go from n_floats*instantiations to instantiations*n_floats
            for i in data_arr:
                ax_n.plot(np.exp(in_x+n_factor), i, linestyle=styles[f], marker=shapes[n], markersize=marksize, color=colors[f], alpha=a)
#                 pz_moment_max[n-1].append(max(i))
        ax_n.set_ylabel(r'$\log[\mathrm{'+moment_names[n]+r'}]$', fontsize=14)
        ax_n.set_ylim(pz_mean_min[n]-1., pz_mean_max[n]+1.)
    ax.set_xscale('log')
    ax.set_xticks(flat_floats)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(np.exp(min(in_x)-0.25), np.exp(max(in_x)+0.25))
    ax.set_xlabel('number of parameters', fontsize=14)
    ax.set_title(dataset_info[dataset_key]['name']+r' data $\log[KLD]$ moments', fontsize=16)
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(loc+'_all.pdf', dpi=250)
    plt.close()

#similar plot with moments of individual pz moment distributions?

def save_nz_metrics(dataset_key, n_gals_use, N_f, nz_klds, stat_name):

    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, stat_name+str(n_gals_use)+dataset_key)
    if os.path.exists(loc+'.hkl'):
        with open(loc+'.hkl', 'r') as nz_file:
        #read in content of list/dict
            nz_stats = hickle.load(nz_file)
    else:
        nz_stats = {}
        nz_stats['N_f'] = []
        for f in formats:
            nz_stats[f] = []

    if N_f not in nz_stats['N_f']:
        nz_stats['N_f'].append(N_f)
        for f in formats:
            nz_stats[f].append([])

    where_N_f = nz_stats['N_f'].index(N_f)

    for f in formats:
        nz_stats[f][where_N_f].append(nz_klds[f])

    with open(loc+'.hkl', 'w') as nz_file:
        hickle.dump(nz_stats, nz_file)

def plot_nz_klds(dataset_key, n_gals_use):

    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, 'nz_klds'+str(n_gals_use)+dataset_key)
    with open(loc+'.hkl', 'r') as nz_file:
        nz_stats = hickle.load(nz_file)
    if len(instantiations) == 10:
        for f in formats:
            if not np.shape(nz_stats[f]) == (4, 10):
                for s in range(len(floats)):
                    nz_stats[f][s] = np.array(np.array(nz_stats[f][s])[:10]).flatten()

    flat_floats = np.array(nz_stats['N_f']).flatten()

    plt.figure(figsize=(5, 5))
    for f in formats:
        data_arr = np.swapaxes(np.array(nz_stats[f]), 0, 1)#turn N_f * instantiations into instantiations * N_f
        n_i = len(data_arr)
        a = 1./len(formats)#1./n_i
        plt.plot([10. * max(flat_floats), 10. * max(flat_floats)], [1., 10.], color=colors[f], alpha=a, label=f, linestyle=styles[f])
        for i in data_arr:
            plt.plot(flat_floats, i, color=colors[f], alpha=a, linestyle=styles[f])
            kld_min.append(min(i))
            kld_max.append(max(i))
    plt.semilogy()
    plt.semilogx()
    plt.xticks(flat_floats, [str(ff) for ff in flat_floats])
    plt.ylim(min(kld_min) / 10., 10. *  max(kld_max))
    plt.xlim(min(flat_floats) / 3., max(flat_floats) * 3.)
    plt.xlabel(r'number of parameters', fontsize=14)
    plt.ylabel(r'KLD', fontsize=14)
    plt.legend(loc='upper right')
    plt.title(r'$\hat{n}(z)$ KLD on '+str(n_gals_use)+' from '+dataset_info[dataset_key]['name']+' mock catalog', fontsize=16)
    plt.savefig(loc+'_all.pdf', dpi=250)
    plt.close()

    plt.figure(figsize=(5, 5))
    a = 1./len(formats)
    for f in formats:
        data_arr = np.swapaxes(np.array(nz_stats[f]), 0, 1)#turn N_f * instantiations into instantiations * N_f
        plt.plot([10. * max(flat_floats), 10. * max(flat_floats)], [1., 10.], color=colors[f], label=f, linestyle=styles[f])
        kld_min.append(np.min(data_arr))
        kld_max.append(np.max(data_arr))
        mean = np.mean(data_arr, axis=0)
        std = np.std(data_arr, axis=0)
        x_cor = np.array([flat_floats[:-1], flat_floats[:-1], flat_floats[1:], flat_floats[1:]])
        y_plus = mean + std
        y_minus = mean - std
        y_cor = np.array([y_minus[:-1], y_plus[:-1], y_plus[1:], y_minus[1:]])
        plt.plot(flat_floats, mean, color=colors[f], linestyle=styles[f])
        plt.fill(x_cor, y_cor, color=colors[f], alpha=a, linewidth=0.)
    plt.semilogy()
    plt.semilogx()
    plt.xticks(flat_floats, [str(ff) for ff in flat_floats])
    plt.ylim(min(kld_min) / 10., 10. *  max(kld_max))
    plt.xlim(min(flat_floats), max(flat_floats))
    plt.xlabel(r'number of parameters', fontsize=14)
    plt.ylabel(r'KLD', fontsize=14)
    plt.legend(loc='upper right')
    plt.title(dataset_info[dataset_key]['name']+r' data $\hat{n}(z)$ KLD', fontsize=16)
    plt.savefig(loc+'_clean.pdf', dpi=250)
    plt.close()

def plot_nz_moments(dataset_key, n_gals_use):

    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, 'nz_moments'+str(n_gals_use)+dataset_key)
    with open(loc+'.hkl', 'r') as nz_file:
        nz_stats = hickle.load(nz_file)
    flat_floats = np.array(nz_stats['N_f']).flatten()
    in_x = np.log(flat_floats)
    a = 1./len(formats)
    shapes = moment_shapes
    marksize = 10

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig, ax = plt.subplots()
    fig.subplots_adjust(right=1.)
    ax_n = ax
    for key in formats:
        ax.plot([-10], [0], color=colors[key], label=key, linestyle=styles[key], linewidth=1)
#     ax.plot([-10], [0], color='k', label='original', linewidth=0.5, alpha=1.)
    for n in range(1, n_moments_use):
        ax.scatter([-10], [0], color='k', alpha=0.5, marker=shapes[n], s=marksize, label=moment_names[n])
        n_factor = 0.1 * (n - 2)
        truth = np.swapaxes(np.array(nz_stats['truth'][n]), 0, 1)
        if n>1:
            ax_n = ax.twinx()
        if n>2:
            ax_n.spines["right"].set_position(("axes", 1. + 0.1 * (n-1)))
            make_patch_spines_invisible(ax_n)
            ax_n.spines["right"].set_visible(True)
        for s in range(len(formats)):
            f = formats[s]
            f_factor = 0.05 * (s - 1)
            data_arr = np.swapaxes(np.array(nz_stats[f][n]), 0, 1) - truth#np.log(np.swapaxes(np.array(nz_stats[f]), 0, 1)[:][:][n])#go from n_floats*instantiations to instantiations*n_floats
            mean = np.mean(data_arr, axis=0).flatten()
            std = np.std(data_arr, axis=0).flatten()
            y_plus = mean + std
            y_minus = mean - std
            y_cor = np.array([y_minus[:-1], y_plus[:-1], y_plus[1:], y_minus[1:]])
            ax_n.plot(np.exp(in_x+n_factor), mean, linestyle=styles[key], marker=shapes[n], markersize=marksize, alpha=2. * a, color=colors[f])
            ax_n.vlines(np.exp(in_x+n_factor), y_minus, y_plus, linewidth=3., alpha=a, color=colors[f])
            nz_mean_max[n] = max(nz_mean_max[n], np.max(y_plus))
            nz_mean_min[n] = min(nz_mean_min[n], np.min(y_minus))
#         data_arr = np.log(np.swapaxes(np.array(nz_stats['truth'][n]), 0, 1))
#         mean = np.mean(data_arr, axis=0).flatten()
#         std = np.std(data_arr, axis=0).flatten()
#         y_plus = mean + std
#         y_minus = mean - std
#         y_cor = np.array([y_minus[:-1], y_plus[:-1], y_plus[1:], y_minus[1:]])
#         ax_n.plot(np.exp(in_x+n_factor), mean, linestyle='-', marker=shapes[n], markersize=marksize, alpha=a, color='k', linewidth=0.5)
#         ax_n.vlines(np.exp(in_x+n_factor), y_minus, y_plus, linewidth=3., alpha=a, color='k')
#         nz_mean_max[n] = max(nz_mean_max[n], np.max(y_plus))
#         nz_mean_min[n] = min(nz_mean_min[n], np.min(y_minus))
#         ax_n.plot(np.exp(in_x+n_factor), np.log(nz_stats['truth'][n]), linestyle='-', marker=shapes[n], markersize=marksize, alpha=a, linewidth=0.5, color='k')
        ax_n.set_ylabel(r'$\Delta\mathrm{'+moment_names[n]+r'}$', fontsize=14)
        ax_n.set_ylim((nz_mean_min[n]-0.1, nz_mean_max[n]+0.1))
    ax.set_xscale('log')
    ax.set_xticks(flat_floats)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(np.exp(min(in_x)-0.25), np.exp(max(in_x)+0.25))
    ax.set_xlabel('number of parameters', fontsize=14)
    ax.set_title(dataset_info[dataset_key]['name']+r' data $\hat{n}(z)$ moments', fontsize=16)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(loc+'_clean.pdf', dpi=250)
    plt.close()

    fig, ax = plt.subplots()
    fig.subplots_adjust(right=1.)
    ax_n = ax
    for key in formats:
        ax_n.plot([-10], [0], color=colors[key], label=key, linestyle=styles[key], linewidth=1)
#     ax.plot([-10], [0], color='k', label='original', linewidth=0.5, alpha=1.)
    for n in range(1, n_moments_use):
        n_factor = 0.1 * (n - 2)
        ax.scatter([-10], [0], color='k', marker=shapes[n], s=marksize, label='moment '+str(n))
        truth = np.swapaxes(np.array(nz_stats['truth'][n]), 0, 1)
        if n>1:
            ax_n = ax.twinx()
        if n>2:
            ax_n.spines["right"].set_position(("axes", 1. + 0.1 * (n-1)))
            make_patch_spines_invisible(ax_n)
            ax_n.spines["right"].set_visible(True)
        for s in range(len(formats)):
            f = formats[s]
            f_factor = 0.05 * (s - 1)
            data_arr = np.swapaxes(np.array(nz_stats[f][n]), 0, 1) - truth
            for i in data_arr:
                ax_n.plot(np.exp(in_x+n_factor), i, linestyle=styles[f], marker=shapes[n], markersize=marksize, color=colors[f], alpha=a)
#                 nz_moment_max[n-1].append(max(i))
        data_arr = np.log(np.swapaxes(np.array(nz_stats['truth'][n]), 0, 1))
#         for i in data_arr:
#             ax_n.plot(np.exp(in_x+n_factor), i, linestyle='-', marker=shapes[n], markersize=marksize, color='k', alpha=a)
# #         ax_n.plot(np.exp(in_x+n_factor), np.log(nz_stats['truth'][n]), linestyle='-', marker=shapes[n], markersize=marksize, alpha=a, linewidth=0.5, color='k')
        ax_n.set_ylabel(r'$\Delta\mathrm{'+moment_names[n]+r'}$', fontsize=14)
        ax_n.set_ylim((nz_mean_min[n]-0.1, nz_mean_max[n]+0.1))
    ax.set_xscale('log')
    ax.set_xticks(flat_floats)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(np.exp(min(in_x)-0.25), np.exp(max(in_x)+0.25))
    ax.set_xlabel('number of parameters', fontsize=14)
    ax.set_title(dataset_info[dataset_key]['name']+r' data $\hat{n}(z)$ moments', fontsize=16)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(loc+'_all.pdf', dpi=250)
    plt.close()

dataset_info = {}
delta = 0.01

dataset_keys = ['mg', 'ss']

for dataset_key in dataset_keys:
    dataset_info[dataset_key] = {}
    if dataset_key == 'mg':
        datafilename = 'bpz_euclid_test_10_3.probs'
        z_low = 0.01
        z_high = 3.51
        nc_needed = 3
        plotname = 'brighter'
        skip_rows = 1
        skip_cols = 1
    elif dataset_key == 'ss':
        datafilename = 'test_magscat_trainingfile_probs.out'
        z_low = 0.005
        z_high = 2.11
        nc_needed = 5
        plotname = 'fainter'
        skip_rows = 1
        skip_cols = 1
    dataset_info[dataset_key]['filename'] = datafilename

    dataset_info[dataset_key]['z_lim'] = (z_low, z_high)
    z_grid = np.arange(z_low, z_high, delta, dtype='float')#np.arange(z_low, z_high + delta, delta, dtype='float')
    z_range = z_high - z_low
    delta_z = z_range / len(z_grid)
    dataset_info[dataset_key]['z_grid'] = z_grid
    dataset_info[dataset_key]['delta_z'] = delta_z

    dataset_info[dataset_key]['N_GMM'] = nc_needed# will be overwritten later
    dataset_info[dataset_key]['name'] = plotname

high_res = 300
color_cycle = np.array([(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178), (213, 94, 0), (204, 121, 167)])/256.
n_plot = len(color_cycle)
n_moments_use = 4
moment_names = ['integral', 'mean', 'variance', 'kurtosis']
moment_shapes = ['o', '*', '+', 'x']

#make this a more clever structure, i.e. a dict
formats = ['quantiles', 'histogram', 'samples']
colors = {'quantiles': 'blueviolet', 'histogram': 'darkorange', 'samples': 'forestgreen'}
styles = {'quantiles': '--', 'histogram': ':', 'samples': '-.'}
stepstyles = {'quantiles': 'dashed', 'histogram': 'dotted', 'samples': 'dashdot'}

iqr_min = [3.5]
iqr_max = [delta]
modes_max = [0]
pz_max = [1.]
nz_max = [1.]
hist_max = [1.]
dist_min = [0.]
dist_max = [0.]
pz_mean_max = -10.*np.ones(n_moments_use)
pz_mean_min = 10.*np.ones(n_moments_use)
kld_min = [1.]
kld_max = [1.]
nz_mean_max = -10.*np.ones(n_moments_use)
nz_mean_min = 10.*np.ones(n_moments_use)

#change all for NERSC

floats = [3, 10, 30, 100]
sizes = [100]#, 1000]
names = dataset_info.keys()
instantiations = range(0, 10)

all_randos = [[np.random.choice(size, n_plot, replace=False) for size in sizes] for name in names]

# the "pipeline"

for n in range(len(names)):
    name = names[n]

    dataset_start = timeit.default_timer()
    print('started '+name)

    pdfs = setup_dataset(name, skip_rows, skip_cols)

    for s in range(len(sizes)):
        size=sizes[s]

        size_start = timeit.default_timer()
        print('started '+name+str(size))

        path = os.path.join(name, str(size))
        if not os.path.exists(path):
            os.makedirs(path)

        n_gals_use = size

        randos = all_randos[n][s]

        for i in instantiations:
            i_start = timeit.default_timer()
            print('started '+name+str(size)+' #'+str(i))

            original = '_original'+str(i)
            pdfs_use = make_instantiation(name, size, pdfs, bonus=original)
#             plot = plot_examples(size, name, bonus=original)

            z_grid = dataset_info[name]['in_z_grid']
            N_comps = dataset_info[name]['N_GMM']

            postfit = '_postfit'+str(i)
            catalog = setup_from_grid(name, pdfs_use, z_grid, N_comps, high_res=high_res, bonus=postfit)
#             plot = plot_examples(size, name, bonus=postfit)

            for n_floats_use in floats:

                float_start = timeit.default_timer()
                print('started '+name+str(size)+' #'+str(i)+' with '+str(n_floats_use))

                (ensembles, pz_klds, metric_moments, pz_moments) = analyze_individual(catalog,
                                                          z_grid,#dataset_info[name]['metric_z_grid'],
                                                          n_floats_use, name, n_moments_use, i=i, bonus=postfit)
                for f in formats:
                    fname = str(n_floats_use)+f+str(i)
#                     plot = plot_examples(size, name, bonus=fname)
#                 plot = plot_individual_kld(size, name, n_floats_use, i=i)
#                 plot = plot_individual_moment(size, name, n_floats_use, i=i)
                save_moments(name, size, n_floats_use, metric_moments, 'pz_kld_moments')
                save_moments(name, size, n_floats_use, pz_moments, 'pz_moments')

                (stack_evals, nz_klds, nz_moments) = analyze_stacked(catalog, ensembles, z_grid,#dataset_info[name]['metric_z_grid'],
                                                     n_floats_use, name, i=i)
#                 plot = plot_estimators(size, name, n_floats_use, i=i)
                save_nz_metrics(name, size, n_floats_use, nz_klds, 'nz_klds')
                save_moments(name, size, n_floats_use, nz_moments, 'nz_moments')

                print('FINISHED '++name+str(size)+' #'+str(i)+' with '+str(n_floats_use)+' in '+str(timeit.default_timer() - float_start))
            print('FINISHED '++name+str(size)+' #'+str(i)+' in '+str(timeit.default_timer() - i_start))
#         plot = plot_pz_metrics(name, size)

#         plot = plot_nz_klds(name, size)
#         plot = plot_nz_moments(name, size)

        print('FINISHED '+name+str(size)+' in '+str(timeit.default_timer() - size_start))

    print('FINISHED '+name+' in '+str(timeit.default_timer() - dataset_start))

#comment out for NERSC
# for name in names:
#     for size in sizes:
#         path = os.path.join(name, str(size))
#         for i in instantiations:
#
#             plot = plot_examples(size, name, bonus='_original'+str(i))
#             plot = plot_examples(size, name, bonus='_postfit'+str(i))
#
#             for n_floats_use in floats:
#
#                 for f in formats:
#                     fname = str(n_floats_use)+f+str(i)
#                     plot = plot_examples(size, name, bonus=fname)
#                 plot = plot_individual_kld(size, name, n_floats_use, i)
#                 plot = plot_individual_moment(size, name, n_floats_use, i)
#                 plot = plot_estimators(size, name, n_floats_use, i)
#
#         plot = plot_pz_metrics(name, size)
#
#         plot = plot_nz_klds(name, size)
#         plot = plot_nz_moments(name, size)
