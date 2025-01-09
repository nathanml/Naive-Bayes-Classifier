from data_util import mean, get_variance, normalize
from math import floor

'''
Returns tuple of likelihood distribution
(b_likelihoods, a_likelihoods) where
likelihood is based on variance.
'''


def extract_features(dataset):
    b_variances = get_variance_distribution(dataset[:10])
    a_variances = get_variance_distribution(dataset[10:])
    return [b_variances, a_variances]


'''
Returns array likelihood distribution based
on variance of the training data provided.
'''


def get_variance_distribution(training_data):
    variances = []
    # Build variances
    for track in training_data:
        mu = mean(track)
        for t in track:
            variances.append(get_variance(t, mu))
    # Create histogram
    histogram = [0 for i in range(200)]
    min_variance = min(variances)
    bin_size = (max(variances) - min(variances)) / 199
    # Bin each variance val
    for variance in variances:
        bin_index = floor((variance - min_variance) / bin_size)
        histogram[bin_index] += 1
    # Return normalized histogram
    return normalize(histogram)
