"""'Nathan Maher Levy"""

from data_util import normalize, get_variance_prob, get_velocity_prob

'''
Write current classification (b_j) to logs.txt file
'''


def log_classification(b_j):
    logs = open("logs.txt", "a")
    if b_j[0] >= b_j[1]:
        logs.write("b, ")
    else:
        logs.write("a, ")
    logs.close()


class NaiveBayesClassifier:
    transition_probs: list  # (b_prob, a_prob) representing probability of previous classification
    features: list  # Array of tuples of probability distributions used to classify (likelihood, a_likelihood)

    '''
        Constructor - No initial features, transition probs set to (0.5,0.5)
    '''

    def __init__(self):
        self.features = []
        self.transition_probs = [0.5, 0.5]

    '''
        Adds likelihood distributions to features used by classifier
    '''

    def add_feature(self, feature):
        self.features.append(feature)

    '''
    Implementation of Recursive Bayesian Estimation Algorithm
    '''

    def classify(self, track):
        path = [track[0]]  # Path is used to calculate the variance based on known velocity vals

        # Calculate probability vector [p_bird, p_airplane)
        b_j = [(get_velocity_prob(track[0], self.features[0][0]) * get_variance_prob(track[0], path,
                                                                                     self.features[1][0]) * 0.5),
               (get_velocity_prob(track[0], self.features[0][1]) * get_variance_prob(track[0], path,
                                                                                     self.features[1][1]) * 0.5)]

        normalize(b_j)  # Normalize

        log_classification(b_j)  # Write current classification to logs
        self.update_transition_probs(b_j)  # Update transition probabilities based on classification

        for i in range(1, len(track)):
            path.append(track[i])
            # [b_prob, a_prob] = velocity_prob * variance_prob * transition_prob
            b_j = [(get_velocity_prob(track[i], self.features[0][0]) * get_variance_prob(track[i], path,
                                                                                         self.features[1][0]) *
                    self.transition_probs[0]),
                   (get_velocity_prob(track[i], self.features[0][1]) * get_variance_prob(track[i], path,
                                                                                         self.features[1][1]) *
                    self.transition_probs[1])]
            normalize(b_j)
            log_classification(b_j)
            self.update_transition_probs(b_j)

        # Return final classification
        if self.transition_probs[0] >= self.transition_probs[1]:
            return "b"
        else:
            return "a"

    '''
    Update transition probabilities based on classification (b_j)
    '''

    def update_transition_probs(self, b_j):
        # If bird
        if b_j[0] >= b_j[1]:
            self.transition_probs[1] = 0.1
            self.transition_probs[0] = 0.9
        else:
            # If airplane
            self.transition_probs[1] = 0.9
            self.transition_probs[0] = 0.1
