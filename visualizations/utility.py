#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np

import matplotlib
matplotlib.use('Agg')

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import rc

x = np.arange(-10, 10, 0.1)
y = np.minimum(x,2)
z = np.minimum(0,x+2)
fig, axes = plt.subplots(1, 2, sharey=True)
fig.set_size_inches(9, 3)

axes[0].plot(x,y)
axes[1].plot(x,z)

for ax in axes:
	ax.set_xlim([-4,5])
	ax.set_ylim([-4,3])
	ax.set_xlabel(u"Alocação ($o_i$)")
	ax.set_ylabel('Utilidade ($u_i$)')


axes[0].set_title("$\\theta_i=2$")
axes[1].set_title('$\\theta_i=-2$')

plt.savefig('../plots/utility.pdf', bbox_inches='tight')
# plt.show()
