'''
=============================
3D allocation (positive only)
=============================

A 3d visualization of the allocation mechanism. Includes only positive actions. Each user is an axis.
'''
import matplotlib
matplotlib.use('TkAgg')

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import numpy as np


# I think normal vector is always (1,1,1,...)/sqrt(N)
# the plane reference point is (R/N,R/N)
# R: Total amount of resources to be distributed
# N: Number of users to receive resources



resources = 6
action = [1,4,2]
credibility = [0,0,0]

bounds = [(0,6),(0,6),(0,8+resources)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

step = 0.25

# Make data.
X = np.arange(bounds[0][0], bounds[0][1] + step, step)
Y = np.arange(bounds[1][0], bounds[1][1] + step, step)
X, Y = np.meshgrid(X, Y)
Z = -X -Y + resources

# discard = np.logical_and(X>0,Y>0)

# X[discard] = np.nan
# Y[discard] = np.nan
Z[Z < 0] = np.nan

print 'plane'
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='#1f77b4', alpha=0.5)



# Y = np.arange(bounds[1][0], bounds[1][1] + step, step)
# X = np.zeros(Y.shape)
# Z = -X - Y + resources
# ax.plot(X,Y,Z, color='black')

Y = np.arange(bounds[1][0], bounds[1][1] + step, step)
Z = 0
X = -Y - Z + resources
ax.plot(X,Y,Z, color='black')

# X = np.arange(bounds[0][0], bounds[0][1] + step, step)
# Y = np.zeros(Y.shape)
# Z = -X - Y + resources
# ax.plot(X,Y,Z, color='black')







Y = np.arange(bounds[1][0], bounds[1][1] + step, step)
X = np.ones(Y.shape)*action[0]
Z = -X - Y + resources
Z[Z < bounds[2][0]] = np.nan
ax.plot(X,Y,Z)

Y = np.arange(bounds[1][0], bounds[1][1] + step, step)
Z = action[2]
X = -Y - Z + resources
X[X < bounds[0][0]] = np.nan
ax.plot(X,Y,Z)

X = np.arange(bounds[0][0], bounds[0][1] + step, step)
Y = np.ones(Y.shape)*action[1]
Z = -X - Y + resources
Z[Z < bounds[2][0]] = np.nan
ax.plot(X,Y,Z)




X = np.arange(bounds[0][0], bounds[0][1] + step, step)
Y = X
Z = X
ax.plot(X,Y,Z)




ax.plot(*[[a] for a in action], markerfacecolor='#d62728', markeredgecolor='w', marker='X', markersize=10, alpha=0.6)
ax.plot(*[[c] for c in credibility], markerfacecolor='black', markeredgecolor='w', marker='o', markersize=10, alpha=0.6)
# ax.plot([1], [1], [1], markerfacecolor='#d62728', markeredgecolor='w', marker='X', markersize=10, alpha=0.6)
# ax.plot([2], [2], [2], markerfacecolor='#d62728', markeredgecolor='w', marker='X', markersize=10, alpha=0.6)

plt.show()
