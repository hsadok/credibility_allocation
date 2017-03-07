
import numpy as np
import scipy.optimize as opt
import unittest


def allocate(a, c, initial_guess):
    def sum_positives(x):
        return sum(i for i in x if i > 0)

    def sum_negatives(x):
        return -sum(i for i in x if i < 0)

    pareto_ref = min(sum_positives(a), sum_negatives(a))

    def sqr_sum(o,c):
        aux = o + c
        return aux.dot(aux)

    def bound_n(x):
        if x < 0:
            return x, 0
        else:
            return x, x

    def bound_p(x):
        if x < 0:
            return x, x
        else:
            return 0, x

    bound = bound_p if sum_positives(a) > sum_negatives(a) else bound_n

    def pareto(x):
        return sum_positives(x) + sum_negatives(x) - 2 * pareto_ref

    bounds = [bound(i) for i in a]

    initial_guess[:] = a  # ignore initial_guess using a as initial_guess

    solution = opt.minimize(
            lambda x: sqr_sum(x, c),
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=({'type': 'eq', 'fun': lambda x: sum(x)},
                         {'type': 'eq', 'fun': pareto}),
            options={'maxiter': 1000, 'disp': False}
        )

    solution = map(lambda x: 0 if np.isnan(x) else int(round(x)), solution.x)
    initial_guess[:] = solution


class TestAllocate(unittest.TestCase):
    def test_lack_resource_int_reputation(self):  # sum a > 0
        desire = [1, 3, 2, -1, -2, 1]
        reputation = [-10, -3, -5, 2, 4, 12]
        output = desire[:]

        allocate(desire, reputation, output)
        self.assertEqual(output, [1, 0, 2, -1, -2, 0])
        self.assertEqual(desire, [1, 3, 2, -1, -2, 1])
        self.assertEqual(reputation, [-10, -3, -5, 2, 4, 12])

        desire = [-1, 3, -2, 1, 2, -1]
        reputation = [10, 3, 5, -2, -4, -12]
        output = desire[:]

        allocate(desire, reputation, output)
        self.assertEqual(output, [-1, 1, -2, 1, 2, -1])

        desire = [-1, 3, -2, 3, 3, -1]
        reputation = [10, 3, 7, -4, -4, -12]
        output = desire[:]

        allocate(desire, reputation, output)
        self.assertEqual(output, [-1, 0, -2, 2, 2, -1])

    def test_lack_resources_float_reputation(self):
        desire = [1, 3, 2, 2, -3, 1, -5, 3, 0]
        reputation = [-6.2, -3.1, -3.1, -2.2, 8.6, 12.2, -4.3, 6.0, -7.9]
        output = desire[:]

        allocate(desire, reputation, output)
        self.assertEqual(output, [1, 3, 2, 2, -3, 0, -5, 0, 0])

    def test_lack_consumer(self):  # sum a < 0
        desire = [1, -3, 2, -1, -2, 1]
        reputation = [-10, -3, -5, 2, 4, 12]
        output = desire[:]

        allocate(desire, reputation, output)
        self.assertEqual(output, [1, -1, 2, -1, -2, 1])

    def test_efficient(self):  # sum a = 0
        desire = [1, -3, 2, -1, -2, 3]
        reputation = [-10, -3, -5, 2, 4, 12]
        output = desire[:]

        allocate(desire, reputation, output)
        self.assertEqual(output, [1, -3, 2, -1, -2, 3])

if __name__ == "__main__":
    unittest.main()
