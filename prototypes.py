#
# NeurIPS 2022 Paper ID 1648 code sumbission
# NeurIPS 2022 Paper Title: Maximum Class Separation as Inductive Bias in One Matrix
#
# Code to generate the matrix with maximum separation.
# A demo code is in LT_CIFAR folder to evalute the matrix on CIFAR-10 and CIFAR-100
# You can also load it directly in your code by calling the create_prototypes() function; without needing to save as a npy file.
#

import sys
import time

from matplotlib import pyplot as plt

sys.setrecursionlimit(10000) #for nr_prototypes>=1000
import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import norm


def create_noisy_prototypes(nr_prototypes, noise_scale=0):
    prototypes = create_prototypes(nr_prototypes)
    if noise_scale != 0:
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=prototypes.shape)
        prototypes = norm(prototypes + noise)
    distances = cdist(prototypes, prototypes)
    avg_dist = distances[~np.eye(*distances.shape, dtype=bool)].mean()
    return prototypes.astype(np.float32), avg_dist

def create_prototypes(nr_prototypes):
    assert nr_prototypes > 0
    prototypes = efficient_V(nr_prototypes)
    assert prototypes.shape == (nr_prototypes, nr_prototypes - 1)
    assert np.all(np.abs(np.sum(np.power(prototypes, 2), axis=1) - 1) <= 1e-6)
    distances = cdist(prototypes, prototypes)
    assert distances[~np.eye(*distances.shape, dtype=bool)].std() <= 1e-3
    return prototypes.astype(np.float32)

def create_prototypes_random(nr_prototypes):
    prototypes = norm(np.random.uniform(size=(nr_prototypes, nr_prototypes - 1)))
    assert prototypes.shape == (nr_prototypes, nr_prototypes - 1)
    assert np.all(np.abs(np.sum(np.power(prototypes, 2), axis=1) - 1) <= 1e-6)
    return prototypes.astype(np.float32)

def V(order):
    if order == 1:
        return np.array([[1, -1]])  # maximally separates on the final axis
    else:
        # on the first call, create the first prototype: [1, 0, 0, ...]
        # subsequent calls get 1*product_of_all_previous_normalization_factors
        col1 = np.zeros((order, 1))
        col1[0] = 1

        # on the nth call, create unnormalized values for the nth axis of all remaining prototypes
        row1 = -1 / order * np.ones((1, order))

        # compute the normalization factor such that [value of current axis (row1), norm_factor * 1] has norm 1
        # due to recursion, we maintain the values in the "future" axes to have norm 1
        # so like (row1, norm_factor1*row2, norm_factor1*norm_factor2*row3, ...)
        normalization = np.sqrt(1 - 1 / (order**2))
        # print("row1", row1)
        # print("normalization", normalization)
        return np.concatenate(  # concats all prototypes
            (
                col1,
                np.concatenate(
                    (
                        row1,
                        normalization * V(order - 1)
                    ),
                    axis=0
                )
            ),
            axis=1
        )


def efficient_V(n_prototypes):
    unnormalized_columns = []
    for i in range(n_prototypes - 1):
        prepend = np.zeros(i)
        n_remaining_axes = n_prototypes - i - 1
        if n_remaining_axes == 1:
            unnormalized_columns.append(np.concatenate((prepend, [1, -1])))
        else:
            append = np.full(n_remaining_axes, -1 / n_remaining_axes)
            unnormalized_columns.append(np.concatenate((prepend, [1], append)))

    prototypes = np.array(unnormalized_columns).T

    compound_norm_factor = 1
    for i in range(1, n_prototypes - 1):
        order = n_prototypes - i
        compound_norm_factor *= np.sqrt(1 - 1 / (order ** 2))
        prototypes[:, i] *= compound_norm_factor

    return prototypes


def benchmark_and_plot(n_min, n_max, n_runs=1000):
    """
    Benchmarks the three functions for n (number of classes) in the range [n_min, n_max].
    For the recursive function V, we call it with order = n-1 and then transpose its output,
    so that all three functions produce an n x (n-1) matrix.
    The function then plots the average execution time for each implementation.
    """
    ns = np.linspace(n_min, n_max, 20, dtype=int)
    times_recursive = []
    times_efficient = []

    for n in ns:
        # Benchmark recursive version (note: transpose to get n x (n-1))
        start = time.perf_counter()
        for _ in range(n_runs):
            res_recursive = V(n - 1).T
        t_rec = (time.perf_counter() - start) / n_runs
        times_recursive.append(t_rec)

        # Benchmark efficient_V
        start = time.perf_counter()
        for _ in range(n_runs):
            res_efficient = efficient_V(n)
        t_eff = (time.perf_counter() - start) / n_runs
        times_efficient.append(t_eff)

        # Verify outputs match (within numerical tolerance)
        assert np.allclose(res_recursive, res_efficient)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(list(ns), times_recursive, marker='o', label='Recursive V')
    plt.plot(list(ns), times_efficient, marker='s', label='Efficient V')
    plt.xlabel('Number of Classes (n)')
    plt.ylabel('Average Time per Call (s)')
    plt.title(f'Benchmark: Average Execution Time over {n_runs} Runs')
    plt.legend()
    plt.grid(True)
    plt.show()


# if __name__ == '__main__':
#     # Adjust the range as desired; here we test for n from 2 up to 20.
#     benchmark_and_plot(5, 1000, n_runs=20)


if __name__ == '__main__':

    nr_classes = 5
    # prototypes = create_prototypes(nr_classes)
    # np.save("prototypes"+str(nr_classes)+".npy", prototypes)
    # prototypes = non_recursive_V(nr_classes - 1)
    prototypes = V(nr_classes-1).T
    # prototypes = efficient_V(nr_classes)
    print(prototypes)

