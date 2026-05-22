#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:00:27 2026

@author: ckadelka
"""

import boolforge as bf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib


colors = matplotlib.cm.Set1
n_min,n_max = 2,7
ns = np.arange(n_min,n_max+1)
n_ns = n_max+1-n_min
res = {'factorials' : [],
       'numbers' : [],
       'avg_sens' : [], 
       'eff_degree' : [],
       'can_strength' : [],
       'layer_structures' : []}

for ii,n in enumerate(ns):
    all_hamming = np.arange(1, 2 ** (n - 1), 2)
    all_abs_bias = 2 * np.abs(all_hamming/2**n - 0.5)

    avg_sens = np.zeros(2 ** (n - 2))
    can_strength = np.zeros_like(avg_sens)
    eff_degree = np.zeros_like(avg_sens)
    layer_structures = []
    factorials = np.zeros_like(avg_sens,dtype=float)
    numbers = np.zeros_like(avg_sens)

    for i, w in enumerate(all_hamming):
        layer = bf.hamming_weight_to_ncf_layer_structure(n, w)
        layer_structures.append(layer)
        factorials[i] = np.prod([math.factorial(l) for l in layer])
        f = bf.random_function(n, layer_structure=layer)
        avg_sens[i] = f.get_average_sensitivity(exact=True, normalized=False)
        can_strength[i] = f.get_canalizing_strength()
        eff_degree[i] = f.get_effective_degree()
        numbers[i] = 2**(n+1) * math.factorial(n) / factorials[i]

    
    res['factorials'].append( factorials )   
    res['numbers'].append( numbers )
    res['avg_sens'].append( avg_sens )
    res['eff_degree'].append( eff_degree )   
    res['can_strength'].append( can_strength )  
    res['layer_structures'].append( layer_structures )    
    
    
    
# f,ax = plt.subplots(figsize=(5,3))
# for ii,n in enumerate(ns):
#     numbers = res['numbers'][ii]
#     factorials = res['factorials'][ii]
#     weights = factorials/np.max(factorials)
#     indices = np.argsort(numbers)
#     ax.loglog(numbers[indices],weights[indices],'x-',label=str(n))
# ax.legend(loc='best',frameon=False,title='Degree')
# ax.set_ylabel('Relative parameter-uniform\nsampling weight of NCFs')
# ax.set_xlabel('Number of NCFs with specific layer structure')
# ax.spines[['top','right']].set_visible(False)
# plt.savefig(f'sampling_prob_vs_numberNCF_n_{n_min}_{n_max}.pdf',bbox_inches='tight')


metrics_x = ['avg_sens','can_strength','eff_degree']
labels_metrics = ['Average sensitivity','Canalizing strength','Effective degree']
f,ax = plt.subplots(len(metrics_x),1,
                    sharey=False,sharex='col',
                    figsize=(5,2*len(metrics_x)))
for i,metric in enumerate(metrics_x):
    mean_x_function_uniform = []
    mean_x_parameter_uniform = []
    for ii,n in enumerate(ns):
        x = res[metric][ii]
        factorials = res['factorials'][ii]
        
        #compute mean metric for both sampling schemes
        weights = 1./factorials
        weights /= sum(weights)
        mean_x_function_uniform.append( np.dot(weights, x) )
        mean_x_parameter_uniform.append( np.mean(x) )
    ax[i].plot(ns,mean_x_function_uniform,'ok-',label='function-uniform')
    ax[i].plot(ns,mean_x_parameter_uniform,'xk--',label='parameter-uniform')

    if i==0:
        ax[i].legend(loc='center',ncol=2,bbox_to_anchor=[0.5,1.15],
                     frameon=False,title='Mean across all NCFs when sampling')
    ax[i].spines[['top','right']].set_visible(False)
    if i==2:
        ax[i].set_xlabel('Number of NCF inputs')
    ax[i].set_ylabel(labels_metrics[i])
plt.savefig(f'impact_summary_n_{n_min}_{n_max}.pdf',bbox_inches='tight')
    


f,ax = plt.subplots(1,len(metrics_x),
                    sharex=False,sharey='row',
                    figsize=(2.5*len(metrics_x),2.5))
for i,metric in enumerate(metrics_x):
    mean_x_function_uniform = []
    mean_x_parameter_uniform = []
    for ii,n in enumerate(ns):
        x = res[metric][ii]
        factorials = res['factorials'][ii]
        
        #compute relative parameter-uniform sampling probability
        y = factorials/np.max(factorials)
        
        ax[i].semilogy(x,y,'x',color=colors(ii))
        # --- regression in log-space ---
        logy = np.log(y)
        coeffs = np.polyfit(x, logy, 1)   # linear fit: log(y) = a*x + b
    
        # fitted curve
        x_fit = np.linspace(min(x), max(x), 500)
        y_fit = np.exp(coeffs[0]*x_fit + coeffs[1])
    
        ax[i].semilogy(x_fit, y_fit, '--', label=str(n),color=colors(ii))  
    if i==1:
        ax[i].legend(loc='center',ncol=5,bbox_to_anchor=[0.5,1.1],
                     frameon=False,title='Number of NCF inputs')
    ax[i].spines[['top','right']].set_visible(False)
    ax[i].set_xlabel(labels_metrics[i])
    if i==0:
        ax[i].set_ylabel('Relative parameter-uniform\nsampling weight of NCFs')
plt.savefig(f'impact_detailed_n_{n_min}_{n_max}.pdf',bbox_inches='tight')





#Bio models analysis
computed_expected_sens = {}

def compositions_bit(n):
    # iterate over all 2^(n-1) ways to place cuts
    for mask in range(1 << (n - 1)):
        comp = []
        current = 1
        for i in range(n - 1):
            if mask & (1 << i):
                comp.append(current)
                current = 1
            else:
                current += 1
        comp.append(current)
        yield comp

def activities_k_canalizing_layers(n, k, layer_structure):
    """
    Expected activities for k-canalizing functions with given layer structure,
    averaged over all cores (doesn't matter if core is allowed to be canalizing or not).
    """
    if k==0:
        return np.ones(n)/2
    
    assert sum(layer_structure) == k
    r = len(layer_structure)

    # --- map variables to layers
    layer_of = []
    for i, ki in enumerate(layer_structure):
        layer_of.extend([i] * ki)

    # --- compute phi
    phi = np.zeros(r + 1)
    for i in range(r - 2, -1, -1):
        exponent_base = np.sum(layer_structure[:i+1])

        s_sum = sum((1/2)**(s + exponent_base)
                    for s in range(layer_structure[i+1]))

        phi[i] = phi[i + 2] + s_sum

    # --- compute psi
    psi = np.zeros(r)
    if n==k: #NCFs
        psi[-1] = 1
    else:
        psi[-1] = 0.5
    for i in range(r - 2, -1, -1):
        psi[i] = 1 - psi[i+1]

    # --- compute activities
    alpha = np.zeros(n)

    # canalizing variables
    
    for j in range(k):
        L = layer_of[j]
        
        alpha[j] = phi[L] + 0.5**(k-1) * psi[L]

    # non-canalizing variables
    if k < n:
        alpha[k:] = 1 / (2**(k+1))

    return alpha

def expected_sens_k_canalizing_layers(n, k, layer_structure):
    if tuple([n,*layer_structure]) in computed_expected_sens:
        return computed_expected_sens[tuple([n,*layer_structure])]
    res = np.sum(activities_k_canalizing_layers(n, k, layer_structure))
    computed_expected_sens[tuple([n,*layer_structure])] = res
    return res

def expected_sens_k(n,k,uniform_over_functions=True):
    if k==0:
        return n/2.
    
    NCF = n==k
    layer_structures = []
    factorials = []
    avg_sens = []
    assert n-k!=1,'no functions with canalizing depth==n-1'
    for layer_structure in compositions_bit(k):
        if not NCF or layer_structure[-1]>1 or n==1:
            layer_structures.append(layer_structure)
            avg_sens.append( expected_sens_k_canalizing_layers(n, k, layer_structure) )
            factorials.append( np.prod([math.factorial(l) for l in layer_structure]) )
    
    factorials = np.array(factorials)
    avg_sens = np.array(avg_sens)
    
    if uniform_over_functions:
        weights = 1./factorials
        weights /= sum(weights)
        return np.dot(weights, avg_sens)
    else:
        return np.mean(avg_sens)

models = bf.get_bio_models_from_repository(simplify_functions=True)
bns = models['BooleanNetworks']
bns = np.array(bns,dtype=bf.BooleanNetwork)

n_models = len(bns)

#Coherence null model significance test
n_null_models = 100
max_size = 20
which_models = np.array([bn.N<=max_size for bn in bns])
coherences = np.zeros((2,sum(which_models),n_null_models))
coherences_bio = np.zeros(sum(which_models))
avg_degree = np.array([np.mean(bn.indegrees[~np.array(list(bn.get_identity_nodes(as_dict=True).values()))]) for bn in bns[which_models]])
for j,bn in enumerate(bns[which_models]):
    coherences_bio[j] = bn.get_attractors_and_robustness_synchronous_exact()['Coherence']
    for i,uniform_over_functions in enumerate([True,False]):
        for k in range(n_null_models):
            try:
                null_model = bf.random_null_model(bn,
                                                  preserve_bias=False,
                                                  preserve_canalizing_depth=True,
                                                  uniform_over_functions=uniform_over_functions)
                coherences[i,j,k] = null_model.get_attractors_and_robustness_synchronous_exact()['Coherence']
            except ValueError: #happens for one network where the rule is degenerate
                continue
        print(i,j)


which = coherences.mean((0,2)) > 0
coherences = coherences[:,which]
coherences_bio = coherences_bio[which]
avg_degree = avg_degree[which]
mean_coherence_func = coherences[0].mean(1)
mean_coherence_parm = coherences[1].mean(1)

c_diff_func = coherences_bio - mean_coherence_func
c_diff_parm = coherences_bio - mean_coherence_parm

# f,ax = plt.subplots()
# im = ax.scatter(avg_degree,
#            mean_coherence_parm - mean_coherence_func,
#            c = mean_coherence_func)
# f.colorbar(im,label = 'Mean coherence function-uniform null models')
# ax.set_xlabel('Average degree')
# ax.set_ylabel('Mean difference in null model coherence\nfunction-uniform minus parameter-uniform')

# f,ax = plt.subplots()
# im = ax.scatter(mean_coherence_parm,
#                 mean_coherence_func,
#                 c = coherences_bio)
# [x1,x2] = ax.get_xlim()
# [y1,y2] = ax.get_ylim()
# ax.plot([0,100],[0,100],'k--')
# ax.set_xlim([min(x1,y1),max(x2,y2)])
# ax.set_ylim([min(x1,y1),max(x2,y2)])
# f.colorbar(im,label = 'Coherence bio model')
# ax.set_xlabel('Mean coherence parameter-uniform null models')
# ax.set_ylabel('Mean coherence function-uniform null models')



# f,ax = plt.subplots()
# im = ax.scatter(coherences_bio - mean_coherence_parm,
#                 coherences_bio - mean_coherence_func,
#                 c = coherences_bio)
# [x1,x2] = ax.get_xlim()
# [y1,y2] = ax.get_ylim()
# ax.plot([-10,100],[-10,100],'k--')
# ax.set_xlim([min(x1,y1),max(x2,y2)])
# ax.set_ylim([min(x1,y1),max(x2,y2)])
# f.colorbar(im,label = 'Coherence bio model')
# ax.set_xlabel('Coherence difference bio models vs\n parameter-uniform null models')
# ax.set_ylabel('Coherence difference bio models vs\n function-uniform null models')






max_degree=15
which_models = np.array([np.max(bn.indegrees)<=max_degree for bn in bns])

observed_avg_sens = []
exp_avg_sens_function_uniform = []
exp_avg_sens_parameter_uniform = []
degree = []
for bn in bns[which_models]:
    observed_avg_sens.append([])
    exp_avg_sens_function_uniform.append([])
    exp_avg_sens_parameter_uniform.append([])
    degree.append([])
    is_identity_node = bn.get_identity_nodes(as_dict=True)
    for i,f in enumerate(bn.F):
        if is_identity_node[i]:
            continue
        n = f.n
        k = f.get_layer_structure()['CanalizingDepth']
        observed_avg_sens[-1].append(
            f.get_average_sensitivity(exact=True,normalized=False)
        )
        exp_avg_sens_function_uniform[-1].append(
            expected_sens_k(n,k,uniform_over_functions=True)
        )
        exp_avg_sens_parameter_uniform[-1].append(
            expected_sens_k(n,k,uniform_over_functions=False)
        )
        degree[-1].append(n)

mean_observed_avg_sens = np.array([np.mean(el) for el in observed_avg_sens])
mean_exp_avg_sens_function_uniform = np.array([np.mean(el) for el in exp_avg_sens_function_uniform])
mean_exp_avg_sens_parameter_uniform = np.array([np.mean(el) for el in exp_avg_sens_parameter_uniform])
mean_degree = np.array([np.mean(el) for el in degree])

# f,ax = plt.subplots()
# im = ax.scatter(mean_exp_avg_sens_function_uniform,
#            mean_observed_avg_sens,c=mean_degree)
# cbar = f.colorbar(im,label='Average degree')
# [x1,x2] = ax.get_xlim()
# [y1,y2] = ax.get_ylim()
# ax.plot([0,100],[0,100],'k--')
# ax.set_xlim([x1,x2])
# ax.set_ylim([y1,y2])
# ax.set_xlabel('Expected function-uniform mean average sensitivity')
# ax.set_ylabel('Observed mean average sensitivity')
# plt.savefig('bio_models1.pdf',bbox_inches='tight')

# f,ax = plt.subplots()
# im = ax.scatter(mean_exp_avg_sens_parameter_uniform,
#            mean_observed_avg_sens,c=mean_degree)
# cbar = f.colorbar(im,label='Average degree')
# [x1,x2] = ax.get_xlim()
# [y1,y2] = ax.get_ylim()
# ax.plot([0,100],[0,100],'k--')
# ax.set_xlim([x1,x2])
# ax.set_ylim([y1,y2])
# ax.set_xlabel('Expected parameter-uniform mean average sensitivity')
# ax.set_ylabel('Observed mean average sensitivity')
# plt.savefig('bio_models2.pdf',bbox_inches='tight')

# f,ax = plt.subplots()
# im = ax.scatter(mean_exp_avg_sens_parameter_uniform,
#         mean_exp_avg_sens_function_uniform,c=mean_degree)
# cbar = f.colorbar(im,label='Average degree')
# [x1,x2] = ax.get_xlim()
# [y1,y2] = ax.get_ylim()
# ax.plot([0,100],[0,100],'k--')
# ax.set_xlim([x1,x2])
# ax.set_ylim([y1,y2])
# ax.set_xlabel('Expected parameter-uniform mean average sensitivity')
# ax.set_ylabel('Expected function-uniform mean average sensitivity')
# plt.savefig('bio_models3.pdf',bbox_inches='tight')

# f,ax = plt.subplots()
# im = ax.scatter(mean_observed_avg_sens,
#                 mean_exp_avg_sens_function_uniform - mean_observed_avg_sens,
#                 c=mean_degree)
# cbar = f.colorbar(im,label='Average degree')
# [x1,x2] = ax.get_xlim()
# [y1,y2] = ax.get_ylim()
# ax.plot([0,100],[0,0],'k--')
# ax.set_xlim([x1,x2])
# ax.set_ylim([y1,y2])
# ax.set_xlabel('Observed mean average sensitivity')
# ax.set_ylabel('Expected-Observed mean average sensitivity\nwhen using function-uniform sampling')
# plt.savefig('bio_models4.pdf',bbox_inches='tight')


# f,ax = plt.subplots()
# im = ax.scatter(mean_observed_avg_sens,
#                 mean_exp_avg_sens_parameter_uniform - mean_observed_avg_sens,
#                 c=mean_degree)
# cbar = f.colorbar(im,label='Average degree')
# [x1,x2] = ax.get_xlim()
# [y1,y2] = ax.get_ylim()
# ax.plot([0,100],[0,0],'k--')
# ax.set_xlim([x1,x2])
# ax.set_ylim([y1,y2])
# ax.set_xlabel('Observed mean average sensitivity')
# ax.set_ylabel('Expected-Observed mean average sensitivity\nwhen using parameter-uniform sampling')
# plt.savefig('bio_models5.pdf',bbox_inches='tight')







#Look instead at actual frequencies of specific type layer structures vs expectation
n_min = 3
n_max = 7
ns = np.arange(n_min,n_max+1)
n_ns = n_max+1-n_min
res = {'factorials' : [],
       'numbers' : [],
       'avg_sens' : [], 
       'eff_degree' : [],
       'can_strength' : [],
       'layer_structures' : []}

for ii,n in enumerate(ns):
    all_hamming = np.arange(1, 2 ** (n - 1), 2)
    all_abs_bias = 2 * np.abs(all_hamming/2**n - 0.5)

    avg_sens = np.zeros(2 ** (n - 2))
    can_strength = np.zeros_like(avg_sens)
    eff_degree = np.zeros_like(avg_sens)
    layer_structures = []
    factorials = np.zeros_like(avg_sens,dtype=float)
    numbers = np.zeros_like(avg_sens)

    for i, w in enumerate(all_hamming):
        layer = bf.hamming_weight_to_ncf_layer_structure(n, w)
        layer_structures.append(tuple(layer))
        factorials[i] = np.prod([math.factorial(l) for l in layer])
        f = bf.random_function(n, layer_structure=layer)
        avg_sens[i] = f.get_average_sensitivity(exact=True, normalized=False)
        can_strength[i] = f.get_canalizing_strength()
        eff_degree[i] = f.get_effective_degree()
        numbers[i] = 2**(n+1) * math.factorial(n) / factorials[i]

    
    res['factorials'].append( factorials )   
    res['numbers'].append( numbers )
    res['avg_sens'].append( avg_sens )
    res['eff_degree'].append( eff_degree )   
    res['can_strength'].append( can_strength )  
    res['layer_structures'].append( layer_structures )    
    




count_np = np.zeros((n_ns,2**(n_max-2)))

degree = []
for bn in bns:
    for i,f in enumerate(bn.F):
        n = f.n
        if n < n_min or n > n_max:
            continue
        k = f.get_layer_structure()['CanalizingDepth']
        if k<n:
            continue
        layer_structure = f.get_layer_structure()['LayerStructure']
        index = res['layer_structures'][n-n_min].index(tuple(layer_structure))
        count_np[n-n_min, index] += 1
        
# colors = ['r','b']
# markers = ['o','x']
# labels = ['function-uniform','parameter-uniform']
# ns = [3,4,5]
# for ii,n in enumerate(ns):
#     f,ax = plt.subplots()
#     avg_sens = res['avg_sens'][ii]
#     y = count_np[ii,:2**(n-2)]
#     y_freq = y/np.sum(y)
    
#     factorials = res['factorials'][ii]
#     exp_y_freq_function_uniform = 1./factorials
#     exp_y_freq_function_uniform /= sum(exp_y_freq_function_uniform)
#     exp_y_freq_parameter_uniform = np.ones(2**(n-2)) / 2**(n-2)
    
#     fc_function_uniform = np.divide(y_freq,exp_y_freq_function_uniform)
#     fc_parameter_uniform = np.divide(y_freq,exp_y_freq_parameter_uniform)
    
#     x = avg_sens
#     for jj,y in enumerate([fc_function_uniform,fc_parameter_uniform]):
#         ax.semilogy(x,y,'o' if jj==0 else 'x',color=colors[jj])
#         # --- regression in log-space ---
#         logy = np.log(y)
#         coeffs = np.polyfit(x, logy, 1)   # linear fit: log(y) = a*x + b
    
#         # fitted curve
#         x_fit = np.linspace(min(x), max(x), 500)
#         y_fit = np.exp(coeffs[0]*x_fit + coeffs[1])
    
#         ax.semilogy(x_fit, y_fit, '--', label=labels[jj],color=colors[jj])

#     ax.set_xlabel('Average sensitivity of NCF')
#     ax.set_ylabel('Fold change in abundance of NCF\n(Observed / Expected)\nwith specific layer structure')
#     ax.legend(loc='best',frameon=False)
#     [x1,x2] = ax.get_xlim()
#     [y1,y2] = ax.get_ylim()
#     ax.plot([0,100],[1,1],'k--')
#     ax.set_xlim([x1,x2])
#     ax.set_ylim([y1,y2])
#     ax.set_title(f'Degree = {n}')
#     plt.savefig(f'bio_models_ncf_fc_{n}.pdf',bbox_inches='tight')



colors = ['r','b']
markers = ['o','x']
labels = ['function-uniform','parameter-uniform']
ns = [3,4,5]
n_ns = len(ns)
f,ax = plt.subplots(n_ns,1,sharex='col',figsize=(5,7.5))
for ii,n in enumerate(ns):
    avg_sens = res['avg_sens'][ii]
    y = count_np[ii,:2**(n-2)]
    y_freq = y/np.sum(y)
    
    factorials = res['factorials'][ii]
    exp_y_freq_function_uniform = 1./factorials
    exp_y_freq_function_uniform /= sum(exp_y_freq_function_uniform)
    exp_y_freq_parameter_uniform = np.ones(2**(n-2)) / 2**(n-2)
    
    fc_function_uniform = np.divide(y_freq,exp_y_freq_function_uniform)
    fc_parameter_uniform = np.divide(y_freq,exp_y_freq_parameter_uniform)
    
    x = avg_sens
    for jj,y in enumerate([fc_function_uniform,fc_parameter_uniform]):
        ax[ii].semilogy(x,y,'o' if jj==0 else 'x',color=colors[jj])
        # --- regression in log-space ---
        logy = np.log(y)
        coeffs = np.polyfit(x, logy, 1)   # linear fit: log(y) = a*x + b
    
        # fitted curve
        x_fit = np.linspace(min(x), max(x), 500)
        y_fit = np.exp(coeffs[0]*x_fit + coeffs[1])
    
        ax[ii].semilogy(x_fit, y_fit, '-' if jj==0 else '--',color=colors[jj],alpha=0.4)

    if ii==2:
        ax[ii].set_xlabel('Average sensitivity of NCF')
    if ii==1:
        ax[ii].set_ylabel('Fold enrichment (biological / expected)')
    ax[ii].spines[['top','right']].set_visible(False)

for ii,n in enumerate(ns):
    [x1,x2] = ax[ii].get_xlim()
    [y1,y2] = ax[ii].get_ylim()

    ax[ii].plot([0,100],[1,1],'k:',alpha=0.5)
    if ii==0:
        for jj,y in enumerate([fc_function_uniform,fc_parameter_uniform]):
            ax[ii].semilogy([0,0], [-1,-1], 'o-' if jj==0 else 'x--', label=labels[jj],color=colors[jj])
        ax[ii].legend(loc='center',ncol=2,bbox_to_anchor=[0.5,1.15],
                     frameon=False,title='Sampling scheme for expectation')
    ax[ii].set_xlim([x1,x2])
    ax[ii].set_ylim([y1,y2])
    ax[ii].text(
        0.97, 0.93, rf'$n={n}$',
        transform=ax[ii].transAxes,
        ha='right',
        va='top'
    )    
ticksets = [
    [0.6,1,2],
    [0.2,1,5],
    [0.1,1,10],
]

from matplotlib.ticker import FixedLocator, FuncFormatter
for ii,n in enumerate(ns):
    ax[ii].yaxis.set_major_locator(FixedLocator(ticksets[ii]))
    ax[ii].yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f'{y:g}')
    )

plt.savefig('bio_models_ncf_fc_all.pdf',bbox_inches='tight')         
