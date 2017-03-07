'''
=============
3D allocation
=============

A 3d visualization of the allocation mechanism. Includes positive and negative actions. Each user is an axis.
'''
import matplotlib
matplotlib.use('TkAgg')

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import numpy as np

bounds = [(-2,2),(-2,2),(-4,4)]

action = [1,2,-2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

step = 0.25

# Make data.
X = np.arange(bounds[0][0], bounds[0][1] + step, step)
Y = np.arange(bounds[1][0], bounds[1][1] + step, step)
X, Y = np.meshgrid(X, Y)
Z = -X -Y

# discard = np.logical_and(X>0,Y>0)

# X[discard] = np.nan
# Y[discard] = np.nan
# Z[Z>2] = np.nan
# Z[Z< -2] = np.nan

print 'plane'
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='#1f77b4', alpha=0.5)



Y = np.arange(bounds[1][0], bounds[1][1] + step, step)
X = np.zeros(Y.shape)
Z = -X - Y
ax.plot(X,Y,Z, color='black')

Y = np.arange(bounds[1][0], bounds[1][1] + step, step)
Z = 0
X = -Y - Z
ax.plot(X,Y,Z, color='black')

X = np.arange(bounds[0][0], bounds[0][1] + step, step)
Y = np.zeros(Y.shape)
Z = -X - Y
ax.plot(X,Y,Z, color='black')







Y = np.arange(bounds[1][0], bounds[1][1] + step, step)
X = np.ones(Y.shape)*action[0]
Z = -X - Y
ax.plot(X,Y,Z)

Y = np.arange(bounds[1][0], bounds[1][1] + step, step)
Z = action[2]
X = -Y - Z
X[X > bounds[0][1]] = np.nan
ax.plot(X,Y,Z)

X = np.arange(bounds[0][0], bounds[0][1] + step, step)
Y = np.ones(Y.shape)*action[1]
Z = -X - Y
ax.plot(X,Y,Z)


# # print 'bounds'
# Y = np.arange(-2, 2 + step, step)
# Z = np.arange(-4, 4 + step, step)
# Y, Z = np.meshgrid(Y, Z)
# X = np.zeros(Z.shape)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='w', alpha=0.3)

# X = np.arange(-2, 2 + step, step)
# Z = np.arange(-4, 4 + step, step)
# X, Z = np.meshgrid(X, Z)
# Y = np.zeros(Z.shape)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='w', alpha=0.3)

# X = np.arange(-2, 2 + step, step)
# Y = np.arange(-2, 2 + step, step)
# X, Y = np.meshgrid(X, Y)
# Z = np.zeros(Y.shape)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='w', alpha=0.3)





# X_sub = np.arange(0, 1 + step, step)
# Y_sub = np.arange(0, 2 + step, step)
# X_sub, Y_sub = np.meshgrid(X_sub, Y_sub)
# Z_sub = -X_sub - Y_sub

# ax.plot_surface(X_sub, Y_sub, Z_sub, rstride=1, cstride=1, color='#ff7f0e')

# X = np.arange(1, 2 + step, step)
# Y = np.arange(0, 2 + step, step)
# X, Y = np.meshgrid(X, Y)
# Z = -X -Y

# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='#1f77b4', alpha=0.7)




ax.plot(*[[a] for a in action], markerfacecolor='#d62728', markeredgecolor='w', marker='X', markersize=10, alpha=0.6)
# ax.plot(*[[a] for a in action], markerfacecolor='black', markeredgecolor='w', marker='X', markersize=10, alpha=0.6)

plt.show()
