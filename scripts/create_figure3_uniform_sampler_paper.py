#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:23:25 2026

@author: ckadelka
"""

import boolforge as bf
import numpy as np
import matplotlib.pyplot as plt

fig_folder = '../figs/'

N = 12
ns = [2,3,4,5,6]
nsim = 10000
uniform_over_functionss=[True,False]

n_ns = len(ns)

coherences = np.zeros((n_ns,2,nsim))
n_attractors = np.zeros((n_ns,2,nsim))
mean_length_attractors = np.zeros((n_ns,2,nsim))
basin_entropies = np.zeros((n_ns,2,nsim))

for i,n in enumerate(ns):
    for j,uniform_over_functions in enumerate(uniform_over_functionss):
        for k in range(nsim):
            bn = bf.random_network(N,n,depth=n,uniform_over_functions=uniform_over_functions)
            info = bn.get_attractors_and_robustness_synchronous_exact()
            n_attractors[i,j,k] = info['NumberOfAttractors']
            coherences[i,j,k] = info['Coherence']
            mean_length_attractors[i,j,k] = np.mean([len(el) for el in info['Attractors']])
            basin_entropies[i,j,k] = bf.get_entropy_of_basin_size_distribution(info['BasinSizes'])

print('mean number attractors',n_attractors.mean(2))
print('mean coherence',coherences.mean(2))
print('mean mean length attractors',mean_length_attractors.mean(2))
print('mean basin entropy',basin_entropies.mean(2))
   
    
print('median number attractors',np.median(n_attractors,2))
print('median coherence',np.median(coherences,2))
print('median mean length attractors',np.median(mean_length_attractors,2))
print('median basin entropy',np.median(basin_entropies,2))      



metric_data = [
    (n_attractors, 'Number of attractors'),
    (mean_length_attractors, 'Mean attractor length'),
    (coherences, 'Coherence'),
    (basin_entropies, 'Basin entropy'),
]

labels = [
    'function-uniform',
    'parameter-uniform'
]

fig, axes = plt.subplots(
    4, 1,
    figsize=(5, 8.5),
    sharex=True
)

for ax, (data, ylabel) in zip(axes, metric_data):

    means = data.mean(axis=2)
    ses = data.std(axis=2, ddof=1) / np.sqrt(data.shape[2])

    for j in range(2):

        ax.plot(
            ns,
            means[:, j],
            'ro-' if j==0 else 'bx--',
            label=labels[j]
        )

        ax.fill_between(
            ns,
            means[:, j] - 1.96 * ses[:, j],
            means[:, j] + 1.96 * ses[:, j],
            alpha=0.2,
            color='r' if j==0 else 'b'
        )

    ax.set_ylabel(ylabel)
    ax.spines[['top','right']].set_visible(False)

axes[0].legend(frameon=False,bbox_to_anchor=[0.5,1.1],
               loc='center',ncol=2,
               title='Mean across random NCF networks when sampling')

axes[-1].set_xlabel('Constant in-degree')
axes[-1].set_xticks(ns)
plt.savefig(fig_folder+'dynamics_implication_uniform_sampler.pdf',bbox_inches='tight')


f,ax = plt.subplots()
ax.plot([0,1])
plt.savefig(fig_folder+'d.pdf',bbox_inches='tight')

            