
# coding: utf-8

# # The Analysis Pipeline
# 
# _Alex Malz (NYU) & Phil Marshall (SLAC)_
# 
# In this notebook we use the "survey mode" machinery to demonstrate how one should choose the optimal parametrization for photo-$z$ PDF storage given the nature of the data, the storage constraints, and the fidelity necessary for a science use case.

# In[ ]:

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[ ]:

from __future__ import print_function
    
import hickle
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import random
import cProfile
import pstats
import StringIO
import timeit
import psutil
import sys
import os
import timeit

import pandas as pd
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import qp
from qp.utils import calculate_kl_divergence as make_kld

# np.random.seed(seed=42)
# random.seed(a=42)


# ## Set-up

# In[ ]:

dataset_info = {}


# There are two datasets available:
# 
# * $10^{5}$ LSST-like mock data provided by Sam Schmidt (UC Davis, LSST)
# * $10^{4}$ Euclid-like mock data provided by Melissa Graham (UW, LSST)

# In[ ]:

# choose one of these:
# dataset_key = 'Euclid'# Melissa Graham's data
# dataset_key = 'LSST'# Sam Schmidt's data
dataset_keys = ['Optical+IR', 'Optical']

for dataset_key in dataset_keys:
    dataset_info[dataset_key] = {}


# Both datasets are fit with BPZ.

# In[ ]:

for dataset_key in dataset_keys:
    if dataset_key == 'Optical+IR':
        datafilename = 'bpz_euclid_test_10_2.probs'
    elif dataset_key == 'Optical':
        datafilename = 'test_magscat_trainingfile_probs.out'
    dataset_info[dataset_key]['filename'] = datafilename


# The data files don't appear to come with information about the native format or metaparameters, but we are told they're evaluations on a regular grid of redshifts with given endpoints and number of parameters.

# In[ ]:

delta = 0.01

for dataset_key in dataset_keys:
    
    if dataset_key == 'Optical+IR':
        z_low = 0.01
        z_high = 3.51
    elif dataset_key == 'Optical':
        z_low = 0.005
        z_high = 2.11
    
    dataset_info[dataset_key]['z_lim'] = (z_low, z_high)

    z_grid = np.arange(z_low, z_high, delta, dtype='float')#np.arange(z_low, z_high + delta, delta, dtype='float')
    z_range = z_high - z_low
    delta_z = z_range / len(z_grid)

    dataset_info[dataset_key]['z_grid'] = z_grid
    dataset_info[dataset_key]['delta_z'] = delta_z


# `qp` cannot currently convert gridded PDFs to histograms or quantiles - we need to make a GMM first, and use this to instantiate a `qp.PDF` object using a `qp.composite` object based on that GMM as `qp.PDF.truth`.  The number of parameters necessary for a qualitatively good fit depends on the characteristics of the dataset. 

# In[ ]:

for dataset_key in dataset_keys:
    
    if dataset_key == 'Optical+IR':
        nc_needed = 3
    elif dataset_key == 'Optical':
        nc_needed = 5
    
    dataset_info[dataset_key]['N_GMM'] = nc_needed


# Let's define some useful quantities.

# In[ ]:

#many_colors = ['red','green','blue','cyan','magenta','yellow']
high_res = 300
n_plot = 5
n_moments_use = 4

#make this a more clever structure, i.e. a dict
formats = ['quantiles', 'histogram', 'samples']
colors = {'quantiles': 'blueviolet', 'histogram': 'darkorange', 'samples': 'forestgreen'}
styles = {'quantiles': '--', 'histogram': ':', 'samples': '-.'}
stepstyles = {'quantiles': 'dashed', 'histogram': 'dotted', 'samples': 'dashdot'}

pz_max = [1.]
nz_max = [1.]
hist_max = [1.]
dist_min = [0.]
dist_max = [0.]
moment_max = [[]] * (n_moments_use - 1)
mean_max = [[]] * (n_moments_use - 1)
kld_min = [1.]
kld_max = [1.]


# ## Analysis
# 
# We want to compare parametrizations for large catalogs, so we'll need to be more efficient.  The `qp.Ensemble` object is a wrapper for `qp.PDF` objects enabling conversions to be performed and metrics to be calculated in parallel.  We'll experiment on a subsample of 100 galaxies.

# In[ ]:

def setup_dataset(dataset_key):#, n_gals_use):
    
    with open(dataset_info[dataset_key]['filename'], 'rb') as data_file:
        lines = (line.split(None) for line in data_file)
        lines.next()
        pdfs = np.array([[float(line[k]) for k in range(1,len(line))] for line in lines])
    
    # sys.getsizeof(pdfs)

#     n_gals_tot = len(pdfs)
#     full_gal_range = range(n_gals_tot)
#     subset = np.random.choice(full_gal_range, n_gals_use, replace=False)#range(n_gals_use)
#     pdfs_use = pdfs[subset]

#     # using the same grid for output as the native format, but doesn't need to be so
#     dataset_info[dataset_key]['in_z_grid'] = dataset_info[dataset_key]['z_grid']
#     dataset_info[dataset_key]['metric_z_grid'] = dataset_info[dataset_key]['z_grid']
    
#     bonus = '_original'
#     path = os.path.join(dataset_key, str(n_gals_use))
#     loc = os.path.join(path, str(n_gals_use)+'from'+dataset_key+'_pzs'+bonus)
#     with open(loc+'.hkl', 'w') as filename:
#         info = {}
#         info['z_grid'] = dataset_info[dataset_key]['in_z_grid']
#         info['pdfs'] = pdfs_use
#         hickle.dump(info, filename)
    
#     return(pdfs_use, bonus)
    return(pdfs)


# In[ ]:

def make_instantiation(dataset_key, n_gals_use, pdfs, bonus=None):
    
    n_gals_tot = len(pdfs)
    full_gal_range = range(n_gals_tot)
    subset = np.random.choice(full_gal_range, n_gals_use, replace=False)#range(n_gals_use)
    pdfs_use = pdfs[subset]

    # using the same grid for output as the native format, but doesn't need to be so
    dataset_info[dataset_key]['in_z_grid'] = dataset_info[dataset_key]['z_grid']
    dataset_info[dataset_key]['metric_z_grid'] = dataset_info[dataset_key]['z_grid']
    
    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, 'pzs'+bonus+str(n_gals_use)+dataset_key)
    with open(loc+'.hkl', 'w') as filename:
        info = {}
        info['randos'] = randos
        info['z_grid'] = dataset_info[dataset_key]['in_z_grid']
        info['pdfs'] = pdfs_use
        hickle.dump(info, filename)
    
    return(pdfs_use)


# In[ ]:

def plot_examples(n_gals_use, dataset_key, bonus=None):
    
    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, 'pzs'+bonus+str(n_gals_use)+dataset_key)
    with open(loc+'.hkl', 'r') as filename:
        info = hickle.load(filename)
        randos = info['randos']
        z_grid = info['z_grid']
        pdfs = info['pdfs']
    
    plt.figure(1)
    for i in range(n_plot):
        data = (z_grid, pdfs[randos[i]])
        data = qp.utils.normalize_integral(qp.utils.normalize_gridded(data))
        pz_max.append(np.max(data))
        plt.plot(data[0], data[1], label=dataset_key+'#'+str(randos[i]))
    plt.xlabel(r'$z$', fontsize=14)
    plt.ylabel(r'$p(z)$', fontsize=14)
    plt.xlim(min(z_grid), max(z_grid))
    plt.ylim(0., max(pz_max))
    plt.title(bonus[1:]+' '+dataset_key+' mock catalog of '+str(n_gals_use), fontsize=16)
    plt.legend()
    
    plt.savefig(loc+'.png', dpi=250)
    plt.close()
    
    plt.figure(2)
    for i in range(n_plot):
        data = (z_grid, pdfs[randos[i]])
        data = qp.utils.normalize_integral(qp.utils.normalize_gridded(data))
        plt.plot(data[0], data[1], label=dataset_key+'#'+str(randos[i]))
    plt.xlabel(r'$z$', fontsize=14)
    plt.ylabel(r'$\log[p(z)]$', fontsize=14)
    plt.semilogy()
    plt.xlim(min(z_grid), max(z_grid))
    plt.ylim(qp.utils.epsilon, max(pz_max))
    plt.title(bonus[1:]+' '+dataset_key+' mock catalog of '+str(n_gals_use), fontsize=16)
    plt.legend()
    
    plt.savefig(loc+'_log.png', dpi=250)
    plt.close()


# We'll start by reading in our catalog of gridded PDFs, sampling them, fitting GMMs to the samples, and establishing a new `qp.Ensemble` object where each meber `qp.PDF` object has `qp.PDF.truth`$\neq$`None`.

# In[ ]:

def setup_from_grid(dataset_key, in_pdfs, z_grid, N_comps, high_res=1000, bonus=None):
    
    #read in the data, happens to be gridded
    zlim = (min(z_grid), max(z_grid))
    N_pdfs = len(in_pdfs)
    
#     plot_examples(N_pdfs, z_grid, pdfs)
    
    print('making the initial ensemble of '+str(N_pdfs)+' PDFs')
    E0 = qp.Ensemble(N_pdfs, gridded=(z_grid, in_pdfs), limits=dataset_info[dataset_key]['z_lim'], vb=True)
    print('made the initial ensemble of '+str(N_pdfs)+' PDFs')
    
    #fit GMMs to gridded pdfs based on samples (faster than fitting to gridded)
    print('sampling for the GMM fit')
    samparr = E0.sample(high_res, vb=False)
    print('took '+str(high_res)+' samples')
    
    print('making a new ensemble from samples')
    Ei = qp.Ensemble(N_pdfs, samples=samparr, limits=dataset_info[dataset_key]['z_lim'], vb=False)
    print('made a new ensemble from samples')
    
    print('fitting the GMM to samples')
    GMMs = Ei.mix_mod_fit(comps=N_comps, vb=False)
    print('fit the GMM to samples')
    
    #set the GMMS as the truth
    print('making the final ensemble')
    Ef = qp.Ensemble(N_pdfs, truth=GMMs, limits=dataset_info[dataset_key]['z_lim'], vb=False)
    print('made the final ensemble')
    
    path = os.path.join(dataset_key, str(N_pdfs))
    loc = os.path.join(path, 'pzs'+bonus+str(N_pdfs)+dataset_key)
    with open(loc+'.hkl', 'w') as filename:
        info = {}
        info['randos'] = randos
        info['z_grid'] = z_grid
        info['pdfs'] = Ef.evaluate(z_grid, using='truth', norm=True, vb=False)[1]
        hickle.dump(info, filename)
    
    return(Ef)


# Next, we compute the KLD between each approximation and the truth for every member of the ensemble.  We make the `qp.Ensemble.kld` into a `qp.PDF` object of its own to compare the moments of the KLD distributions for different parametrizations.

# In[ ]:

def analyze_individual(E, z_grid, N_floats, dataset_key, N_moments=4, i=None):
    zlim = (min(z_grid), max(z_grid))
    z_range = zlim[-1] - zlim[0]
    delta_z = z_range / len(z_grid)
    
    Eq, Eh, Es = E, E, E
    inits = {}
    for f in formats:
        inits[f] = {}
        for ff in formats:
            inits[f][ff] = None
            
    qstart = timeit.default_timer()
    print('performing quantization')
    inits['quantiles']['quantiles'] = Eq.quantize(N=N_floats, vb=False)
    print('finished quantization at '+str(timeit.default_timer() - qstart))
    hstart = timeit.default_timer()
    print('performing histogramization')
    inits['histogram']['histogram'] = Eh.histogramize(N=N_floats, binrange=zlim, vb=False)
    print('finished histogramization at '+str(timeit.default_timer() - hstart))
    sstart = timeit.default_timer()
    print('performing sampling')
    inits['samples']['samples'] = Es.sample(samps=N_floats, vb=False)
    print('finished sampling at '+str(timeit.default_timer() - sstart))
        
    print('making the approximate ensembles')
    Eo = {}
    for f in formats:
        Eo[f] = qp.Ensemble(E.n_pdfs, truth=E.truth, 
                            quantiles=inits[f]['quantiles'], 
                            histogram=inits[f]['histogram'],
                            samples=inits[f]['samples'], 
                            limits=dataset_info[dataset_key]['z_lim'])
        bonus = '_'+str(n_floats_use)+f+'_('+str(i)+')'
        path = os.path.join(dataset_key, str(n_gals_use))
        loc = os.path.join(path, 'pzs'+bonus+str(n_gals_use)+dataset_key)
        with open(loc+'.hkl', 'w') as filename:
            info = {}
            info['randos'] = randos
            info['z_grid'] = z_grid
            info['pdfs'] = Eo[f].evaluate(z_grid, using=f, norm=True, vb=False)[1]
            hickle.dump(info, filename)
    print('made the approximate ensembles')
    
    print('calculating the individual metrics')
    metric_start = timeit.default_timer()
    klds, metrics, moments = {}, {}, {}
    
    for key in Eo.keys():
        print('starting '+key)
        klds[key] = Eo[key].kld(using=key, limits=zlim, dx=delta_z)
        samp_metric = qp.PDF(samples=klds[key])
        gmm_metric = samp_metric.mix_mod_fit(n_components=dataset_info[dataset_key]['N_GMM'], 
                                             using='samples', vb=False)
        metrics[key] = qp.PDF(truth=gmm_metric)
        moments[key] = []
        for n in range(N_moments+1):
            moments[key].append([qp.utils.calculate_moment(metrics[key], n,
                                                          using=key, 
                                                          limits=zlim, 
                                                          dx=delta_z, 
                                                          vb=False)])
        print('finished with '+key)
    print('calculated the individual metrics in '+str(timeit.default_timer() - metric_start))

    path = os.path.join(dataset_key, str(E.n_pdfs))
    loc = os.path.join(path, str(N_floats)+'kld_hist'+str(n_gals_use)+dataset_key+str(i))
    with open(loc+'.hkl', 'w') as filename:
        info = {}
        info['z_grid'] = z_grid
        info['N_floats'] = N_floats
        info['pz_klds'] = klds
        hickle.dump(info, filename)
    
    return(Eo, klds, moments)


# In[ ]:

def plot_individual(n_gals_use, dataset_key, N_floats, i):
    
    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, str(N_floats)+'kld_hist'+str(n_gals_use)+dataset_key+str(i))
    with open(loc+'.hkl', 'r') as filename:
        info = hickle.load(filename)
        z_grid = info['z_grid']
        N_floats = info['N_floats']
        pz_klds = info['pz_klds']
    
    plt.figure()
    plot_bins = np.linspace(-3., 3., 20)
    a = 1./len(formats)
    for key in pz_klds.keys():
        logdata = qp.utils.safelog(pz_klds[key])
        kld_hist = plt.hist(logdata, color=colors[key], alpha=a, histtype='stepfilled', edgecolor='k',
             label=key, normed=True, bins=plot_bins, linestyle=stepstyles[key], ls=stepstyles[key], lw=3)
        hist_max.append(max(kld_hist[0]))
        dist_min.append(min(logdata))
        dist_max.append(max(logdata))
    plt.legend()
    plt.ylabel('frequency', fontsize=14)
    plt.xlabel(r'$\log[KLD]$', fontsize=14)
    plt.xlim(min(dist_min), max(dist_max))
    plt.ylim(0., max(hist_max))
    plt.title('KLD distribution of '+str(n_gals_use)+' from '+dataset_key+r' with $N_{f}='+str(N_floats)+r'$', fontsize=16)
    plt.savefig(loc+'.png', dpi=250)
    plt.close()


# Finally, we calculate metrics on the stacked estimator $\hat{n}(z)$ that is the average of all members of the ensemble.

# In[ ]:

def analyze_stacked(E0, E, z_grid, n_floats_use, dataset_key, i=None):
    
    zlim = (min(z_grid), max(z_grid))
    z_range = zlim[-1] - zlim[0]
    delta_z = z_range / len(z_grid)
    
    print('stacking the ensembles')
    stack_start = timeit.default_timer()
    stacked_pdfs, stacks = {}, {}
    for key in formats:
        stacked_pdfs[key] = qp.PDF(gridded=E[key].stack(z_grid, using=key, 
                                                        vb=False)[key])
        stacks[key] = stacked_pdfs[key].evaluate(z_grid, using='gridded', norm=True, vb=False)[1]
    
    stacked_pdfs['truth'] = qp.PDF(gridded=E0.stack(z_grid, using='truth', 
                                                    vb=False)['truth'])
    
    stacks['truth'] = stacked_pdfs['truth'].evaluate(z_grid, using='gridded', norm=True, vb=False)[1]
    print('stacked the ensembles in '+str(timeit.default_timer() - stack_start))
    
    print('calculating the metrics')
    metric_start = timeit.default_timer()
    klds = {}
    for key in formats:
        klds[key] = qp.utils.calculate_kl_divergence(stacked_pdfs['truth'],
                                                     stacked_pdfs[key], 
                                                     limits=zlim, dx=delta_z)
    print('calculated the metrics in '+str(timeit.default_timer() - metric_start))
    
    path = os.path.join(dataset_key, str(E0.n_pdfs))
    loc = os.path.join(path, str(n_floats_use)+'nz_comp'+str(n_gals_use)+dataset_key+str(i))
    with open(loc+'.hkl', 'w') as filename:
        info = {}
        info['z_grid'] = z_grid
        info['stacks'] = stacks
        info['klds'] = klds
        hickle.dump(info, filename)
    
    return(stacked_pdfs, klds)


# In[ ]:

def plot_estimators(n_gals_use, dataset_key, n_floats_use, i=None):
    
    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, str(n_floats_use)+'nz_comp'+str(n_gals_use)+dataset_key+str(i))
    with open(loc+'.hkl', 'r') as filename:
        info = hickle.load(filename)
        z_grid = info['z_grid']
        stacks = info['stacks']
        klds = info['klds']
    
    plt.figure()
    plt.plot(z_grid, stacks['truth'], color='black', lw=4, alpha=0.3, label='truth')
    nz_max.append(max(stacks['truth']))
    for key in formats:
        nz_max.append(max(stacks[key]))
        plt.plot(z_grid, stacks[key], label=key+r' KLD='+str(klds[key]), color=colors[key], linestyle=styles[key])
    plt.xlabel(r'$z$', fontsize=14)
    plt.ylabel(r'$\hat{n}(z)$', fontsize=14)
    plt.xlim(min(z_grid), max(z_grid))
    plt.ylim(0., max(nz_max))
    plt.legend()
    plt.title(r'$\hat{n}(z)$ for '+str(n_gals_use)+r' from '+dataset_key+r' with $N_{f}='+str(n_floats_use)+r'$', fontsize=16)
    plt.savefig(loc+'.png', dpi=250)
    plt.close()


# We save the data so we can remake the plots later without running everything again.

# ## Scaling
# 
# We'd like to do this for many values of $N_{f}$ as well as larger catalog subsamples, repeating the analysis many times to establish error bars on the KLD as a function of format, $N_{f}$, and dataset.  The things we want to plot across multiple datasets/number of parametes are:
# 
# 1. KLD of stacked estimator, i.e. `N_f` vs. `nz_output[dataset][format][instantiation][KLD_val_for_N_f]`
# 2. moments of KLD of individual PDFs, i.e. `n_moment, N_f` vs. `pz_output[dataset][format][n_moment][instantiation][moment_val_for_N_f]`
# 
# So, we ned to make sure these are saved!

# We want to plot the moments of the KLD distribution for each format as $N_{f}$ changes.

# In[ ]:

def save_pz_metrics(dataset_key, n_gals_use, N_f, metric_moments):

    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, 'pz_klds'+str(n_gals_use)+dataset_key)
    
    if os.path.exists(loc+'.hkl'):
        with open(loc+'.hkl', 'r') as pz_file:
        #read in content of list/dict
            pz_stats = hickle.load(pz_file)
    else:
        pz_stats = {}
        pz_stats['N_f'] = []
        for f in formats:#change this name to formats
            pz_stats[f] = []
            for m in range(n_moments_use + 1):
                pz_stats[f].append([])

    if N_f not in pz_stats['N_f']:
        pz_stats['N_f'].append(N_f)
        for f in formats:
            for m in range(n_moments_use + 1):
                pz_stats[f][m].append([])
        
    where_N_f = pz_stats['N_f'].index(N_f)
        
    for f in formats:
        for m in range(n_moments_use + 1):
            pz_stats[f][m][where_N_f].append(metric_moments[f][m])

    with open(loc+'.hkl', 'w') as pz_file:
        hickle.dump(pz_stats, pz_file)


# In[ ]:

def plot_pz_metrics(dataset_key, n_gals_use):
# trying really hard to make this colorblind-readable but still failing

    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, 'pz_klds'+str(n_gals_use)+dataset_key)
    with open(loc+'.hkl', 'r') as pz_file:
        pz_stats = hickle.load(pz_file)
        
    flat_floats = np.array(pz_stats['N_f']).flatten()

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    shapes = ['*','+','x']#,'v','^','<','>']
    marksize = 50
    a = 1./len(formats)
    
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=1.)
    ax_n = ax
    for key in formats:
        ax_n.plot([-1], [0], color=colors[key], label=key, linestyle=styles[key], linewidth=1)

    for n in range(1, 4):
        ax.scatter([-1], [0], color='k', marker=shapes[n-1], s=marksize, label='moment '+str(n))
        if n>1:
            ax_n = ax.twinx()
        if n>2:
            ax_n.spines["right"].set_position(("axes", 1. + 0.1 * (n-1)))
            make_patch_spines_invisible(ax_n)
            ax_n.spines["right"].set_visible(True)
        for f in formats:
            data_arr = np.swapaxes(np.array(pz_stats[f][n]), 0, 1)#go from n_floats*instantiations to instantiations*n_floats
            for i in data_arr:#next try plot with marker and linewidth/linestyle keywords
                ax_n.scatter(flat_floats, i, marker=shapes[n-1], s=marksize, color=colors[f], alpha=a)#, 
                             # linewidth=1, linestyle=styles[f], edgecolor='k')
#                 ax_n.scatter(flat_floats, i, marker=shapes[n-1], s=marksize, color='None',
#                              linewidth=2, linestyle=styles[f], edgecolor='k', alpha=1.)
                moment_max[n-1].append(max(i))
        ax_n.set_ylabel('moment '+str(n), fontsize=14)
        ax_n.set_ylim(0., max(moment_max[n-1]))
    ax.set_xlim(min(flat_floats) - 10**int(np.log10(min(flat_floats))), max(flat_floats) + 10**int(np.log10(max(flat_floats))))
    ax.semilogx()
    ax.set_xlabel('number of parameters', fontsize=14)
    ax.set_title('KLD moments on '+str(n_gals_use)+' from '+dataset_key, fontsize=16)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(loc+'.png', dpi=250)
    plt.close()
    
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=1.)
    ax_n = ax
#     jitters = {}
#     factors = {'quantiles':-0.1, 'histogram':0., 'samples':0.1}
    for key in formats:
        ax_n.plot([-1], [0], color=colors[key], label=key, linewidth=1)
#         jitters[key] = factors[key] * np.sqrt(flat_floats)
    for n in range(1, 4):
        ax.scatter([-1], [0], color='k', marker=shapes[n-1], s=marksize, label='moment '+str(n))
        if n>1:
            ax_n = ax.twinx()
        if n>2:
            ax_n.spines["right"].set_position(("axes", 1. + 0.1 * (n-1)))
            make_patch_spines_invisible(ax_n)
            ax_n.spines["right"].set_visible(True)
        for f in formats:
            data_arr = np.swapaxes(np.array(pz_stats[f][n]), 0, 1)#go from n_floats*instantiations to instantiations*n_floats
            mean = np.mean(data_arr, axis=0).flatten()
            std = np.std(data_arr, axis=0).flatten()
#             x_cor = np.array([flat_floats[:-1], flat_floats[:-1], flat_floats[1:], flat_floats[1:]])
#             y_plus = mean + std
#             y_minus = mean - std
#             y_cor = np.array([y_minus[:-1], y_plus[:-1], y_plus[1:], y_minus[1:]])
            ax_n.scatter(flat_floats, mean, marker=shapes[n-1], s=marksize, alpha=2*a, color=colors[f])
            ax_n.errorbar(flat_floats, mean, yerr=std, color=colors[f], alpha=2*a, capsize=5, elinewidth=1, linewidth=0., visible=True)
#             ax_n.fill(x_cor, y_cor, color=colors[f], alpha=a, linewidth=0.)
            mean_max[n-1].append(np.max(mean+std))
        ax_n.set_ylabel('moment '+str(n), fontsize=14)
        ax_n.set_ylim(0., np.max(np.array(mean_max[n-1])))
    ax.set_xlim(min(flat_floats)/3., max(flat_floats)*3.)
    ax.semilogx()
    ax.set_xlabel('number of parameters', fontsize=14)
    ax.set_title('KLD moments on '+str(n_gals_use)+' from '+dataset_key, fontsize=16)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(loc+'_clean.png', dpi=250)
    plt.close()


# We want to plot the KLD on $\hat{n}(z)$ for all formats as $N_{f}$ changes.  We want to repeat this for many subsamples of the catalog to establush error bars on the KLD values.

# In[ ]:

def save_nz_metrics(dataset_key, n_gals_use, N_f, nz_klds):
    
    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, 'nz_kld'+str(n_gals_use)+dataset_key)
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


# In[ ]:

def plot_nz_metrics(dataset_key, n_gals_use):
    
    path = os.path.join(dataset_key, str(n_gals_use))
    loc = os.path.join(path, 'nz_kld'+str(n_gals_use)+dataset_key)
    with open(loc+'.hkl', 'r') as nz_file:
        nz_stats = hickle.load(nz_file)

    flat_floats = np.array(nz_stats['N_f']).flatten()
    
    plt.figure(figsize=(5, 5))

    for f in formats:
#     mu = np.mean(np.array(nz_stats[dataset_key][f]), axis=0)
#     sigma = np.std(np.array(nz_stats[dataset_key][f]), axis=0)
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
    plt.ylim(min(kld_min) / 10., 10. *  max(kld_max))
    plt.xlim(min(flat_floats) / 3., max(flat_floats) * 3.)
    plt.xlabel(r'number of parameters', fontsize=14)
    plt.ylabel(r'KLD', fontsize=14)
    plt.legend(loc='upper right')
    plt.title(r'$\hat{n}(z)$ KLD on '+str(n_gals_use)+' from '+dataset_key, fontsize=16)

    plt.savefig(loc+'.png', dpi=250)
    plt.close()

    plt.figure(figsize=(5, 5))
    a = 1./len(formats)
    for f in formats:
#     mu = np.mean(np.array(nz_stats[dataset_key][f]), axis=0)
#     sigma = np.std(np.array(nz_stats[dataset_key][f]), axis=0)
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
    plt.ylim(min(kld_min) / 10., 10. *  max(kld_max))
    plt.xlim(min(flat_floats) / 3., max(flat_floats) * 3.)
    plt.xlabel(r'number of parameters', fontsize=14)
    plt.ylabel(r'KLD', fontsize=14)
    plt.legend(loc='upper right')
    plt.title(r'$\hat{n}(z)$ KLD on '+str(n_gals_use)+' from '+dataset_key, fontsize=16)

    plt.savefig(loc+'_clean.png', dpi=250)
    plt.close()


# # Okay, now all I have to do is have this loop over both datasets, number of galaxies, number of floats, and instantiations!
# 
# Note: It takes about 5 minutes per \# floats considered for 100 galaxies, and about 40 minutes per \# floats for 1000 galaxies.  (So, yes, it scales more or less as expected!)

# In[ ]:

floats = [3, 10, 30, 100]
sizes = [100]#, 1000, 10000]
names = dataset_info.keys()#['Optical', 'Optical+IR']
instantiations = range(0, 2)

#many_colors = ['red','green','blue','cyan','magenta','yellow']
high_res = 300
n_plot = 5
n_moments_use = 4

#make this a more clever structure, i.e. a dict
formats = ['quantiles', 'histogram', 'samples']
colors = {'quantiles': 'blueviolet', 'histogram': 'darkorange', 'samples': 'forestgreen'}
styles = {'quantiles': '--', 'histogram': ':', 'samples': '-.'}
stepstyles = {'quantiles': 'dashed', 'histogram': 'dotted', 'samples': 'dashdot'}

pz_max = [1.]
nz_max = [1.]
hist_max = [1.]
dist_min = [0.]
dist_max = [0.]
moment_max = [[]] * (n_moments_use - 1)
mean_max = [[]] * (n_moments_use - 1)
kld_min = [1.]
kld_max = [1.]

randos = [np.random_choice(size, (len(names), n_plot), replace=False) for size in sizes]


# The "pipeline" is a bunch of nested `for` loops because `qp.Ensemble` makes heavy use of multiprocessing.  Doing multiprocessing within multiprocessing may or may not cause problems, but I am certain that it makes debugging a nightmare.
# 
# Okay, without further ado, let's do it!

# In[ ]:

# the "pipeline"

for n in range(len(names)):
    name = names[n]
    
    dataset_start = timeit.default_timer()
    print('started '+name)
    
    pdfs = setup_dataset(name)
    
    for s in range(len(sizes)):
        size=sizes[s]
        
        size_start = timeit.default_timer()
        print('started '+str(size)+name)
        
        path = os.path.join(name, str(size))
        if not os.path.exists(path):
            os.makedirs(path)
        
        n_gals_use = size
        
        randos = randos[s][n]#np.random.choice(size, n_plot, replace=False)
        
        for i in instantiations:
        
            original = '_original_('+str(i)+')'
            pdfs_use = make_instantiation(name, size, pdfs, bonus=original)
#             plot = plot_examples(size, name, bonus=original)
        
            z_grid = dataset_info[name]['in_z_grid']
            N_comps = dataset_info[name]['N_GMM']
        
            postfit = '_post-fit_('+str(i)+')'
            catalog = setup_from_grid(name, pdfs_use, z_grid, N_comps, high_res=high_res, bonus=postfit)
#             plot = plot_examples(size, name, bonus=postfit)
        
            for n_floats_use in floats:
            
                float_start = timeit.default_timer()
                print('started '+str(size)+name+str(n_floats_use)+'\#'+str(i))
        
                (ensembles, pz_klds, metric_moments) = analyze_individual(catalog, 
                                                          z_grid,#dataset_info[name]['metric_z_grid'], 
                                                          n_floats_use, name, n_moments_use, i=i)
                for f in formats:
                    fname = '_'+str(n_floats_use)+f+'_('+str(i)+')'
#                     plot = plot_examples(size, name, bonus=fname)
#                 plot = plot_individual(size, name, n_floats_use, i=i)
                save_pz_metrics(name, size, n_floats_use, metric_moments)
            
                (stack_evals, nz_klds) = analyze_stacked(catalog, ensembles, z_grid,#dataset_info[name]['metric_z_grid'], 
                                                     n_floats_use, name, i=i)
#                 plot = plot_estimators(size, name, n_floats_use, i=i)
                save_nz_metrics(name, size, n_floats_use, nz_klds)
            
                print('finished '+str(size)+name+str(n_floats_use)+' in '+str(timeit.default_timer() - float_start))
        
#         plot = plot_pz_metrics(name, size)
        
#         plot = plot_nz_metrics(name, size)
        
        print('finished '+str(size)+name+' in '+str(timeit.default_timer() - size_start))
        
    print('finished '+name+' in '+str(timeit.default_timer() - dataset_start))


# Remake the plots to share axes.

# In[ ]:

for name in names:
    for size in sizes:
        path = os.path.join(name, str(size))
        for i in instantiations:
            
            plot = plot_examples(size, name, bonus='_original_('+str(i)+')')
        
            plot = plot_examples(size, name, bonus='_post-fit_('+str(i)+')')
            
            for n_floats_use in floats:
            
                for f in formats:
                    fname = '_'+str(n_floats_use)+f+'_('+str(i)+')'
                    plot = plot_examples(size, name, bonus=fname)
                plot = plot_individual(size, name, n_floats_use, i)
            
                plot = plot_estimators(size, name, n_floats_use, i)
            
        plot = plot_pz_metrics(name, size)
        
        plot = plot_nz_metrics(name, size)


# 

# In[ ]:




# In[ ]:



