import timeit


print '\nLexicographic Allocation'
print timeit.timeit('allocate(desire, reputation, output)', setup='from allocate import allocate; desire = [1, 3, 2, 2, -3, 1, -5, 3, 0]; reputation = [-6.2, -3.1, -3.1, -2.2, 8.6, 12.2, -4.3, 6.0, -7.9]; output = [0]*9', number=10000)
print '-------------------------'

print '\nOptimization Allocation'
print timeit.timeit('allocate(desire, reputation, output)', setup='from optimization_allocation import allocate; desire = [1, 3, 2, 2, -3, 1, -5, 3, 0]; reputation = [-6.2, -3.1, -3.1, -2.2, 8.6, 12.2, -4.3, 6.0, -7.9]; output = desire[:]', number=10000)
