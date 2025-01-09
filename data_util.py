import numpy
from math import floor

'''
Returns mean of dataset
'''

def mean(dataset):
    return sum(dataset) // len(dataset)


'''
Returns variance based on value and mean
'''


def get_variance(x, mu):
    return numpy.square(x - mu)


'''
Returns normalized input
'''


def normalize(probability_vector):
    return [p / sum(probability_vector) for p in probability_vector]


'''
Looks up likelihood based on the velocity distribution
provided (bin index = value * 2)
'''


def get_velocity_prob(val, dist):
    return dist[floor(val * 2)]


'''
Looks up likelihood based on the variance distribution
provided. Calculates variance of the val given the
current set provided and looks up that variance in 
the likelihood distribution provided.
'''


def get_variance_prob(val, current_set, dist):
    variance = get_variance(val, mean(current_set))
    # Get Bin
    bin_size = max(dist) - min(dist) / 199
    bin_index = 0
    i = min(dist)
    while i < variance and i < max(dist):
        bin_index += 1
        i += bin_size
    return dist[bin_index]
