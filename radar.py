"""'Nathan Maher Levy"""

from nbc import NaiveBayesClassifier
from feature_engineering import extract_features
import os

'''
Formatting of Data Files
'''
import numpy

SOLUTION = "bbbaabaaab"

'''
Calls extract function to get formatted datasets
'''


def setup():
    # Extract likelihood distribution of bird/plane
    likelihoods = extract_from_file("likelihood.txt")
    training = extract_from_file("dataset.txt")
    testing = extract_from_file("testing.txt")
    return likelihoods, training, testing


'''
Extracts data (array of floats) from .txt file provided
'''


def extract_from_file(filename):
    file = open(filename, "r")
    dataset = []
    for line in file:
        dataset.append([float(s) for s in line.split() if not numpy.isnan(float(s))])
    return dataset


'''
Delete old logs
'''


def initialize_logs():
    try:
        os.remove("logs.txt")
    except FileNotFoundError:
        pass


'''
For printing breaks between each measured object in testing set
'''


def log():
    logs = open("logs.txt", "a")
    logs.write("\n=================================================\n")


'''
Execution begins here
'''
initialize_logs()
(velocities, training_data, testing_data) = setup()
nbc = NaiveBayesClassifier()  # instantiate classifier
nbc.add_feature(velocities)  # Add velocities
variances = extract_features(training_data)  # Extract variances
nbc.add_feature(variances)  # Add variances

# Iterate through testing data
correct = 0
for i in range(len(testing_data)):
    track = testing_data[i]
    log()
    classification = nbc.classify(track)  # Classify each track in testing set
    print("O_", i + 1, " : ", classification)  # Print classification
    # If correct, update counter
    if classification == SOLUTION[i]:
        correct += 1
print("Accuracy : ", correct / 10 * 100, "%")  # Print percentage correct
