# Naive Bayesian Classification
## Instructions to run and test
From the root directory, run the following command in the 
command line:
```
python radar.py
```
Running this command will trigger the classification algorithm
on the testing dataset. Final classifications are printed to the 
console, while the classification of each data sample is printed
to the file logs.txt.

## Design
### Implementation
This program implements the recursive bayesian classification 
algorithm to classify each track in testing. At each velocity
measurement in the track, the algorithm calculates and normalizes
a probability vector (b_prob, a_prob), representing the probability
that the measured object is a bird or plane respectively.

These probabilities are calculated by multiplying the emission
probability by the transition probability. Emission probability
uses the probability distribution for velocities (provided in 
likelihood.txt) to calculate the velocity probability. It also 
uses an additional feature to improve these velocity measurements.

The transition probabilities are:

𝑃(𝐶_𝑡+1=𝑏𝑖𝑟𝑑 | 𝐶_𝑡=𝑏𝑖𝑟𝑑)=0.9

𝑃(𝐶𝑡+1=𝑎𝑖𝑟𝑝𝑙𝑎𝑛𝑒 | 𝐶𝑡=𝑎𝑖𝑟𝑝𝑙𝑎𝑛𝑒)=0.9

### Additional Feature
Prior to classification, an additional feature (variance) is extracted from
dataset.txt. For each measurement in each track in dataset.txt, the variance
is calculated (based only on the set of variances measured so far), and binned
to generate a probability distribution of the variance measurements for birds
and planes. This likelihood distribution is then used in addition to the velocity 
distribution to calculate the emission probability for each sample.

### Project Structure
This project is split into 4 python files:
```
radar.py: File for running the project. Extracts data from
           the provided files, initializes logs. Extracts
           features from the dataset provided and initializes 
           the classifier with those features. Calls
           the classify function and measures the performance.
nbc.py: Implements recursive bayesian estimation algorithm
feature_engineering.py: Functions for extracting new feature (variance).
data_util.py: Implements shared data operations (mean etc.)
```

### Naive Bayes Classifier

In nbc.py, I've created a class NaiveBayesClassifier in which the algorithm
is implemented. This class has 2 attributes: a transition probability 
vector representing (p_bird, p_airplane) of the last classification, and
a features array, holding the likelihood distributions of the various
features that will be used. In radar.py, this class is instantiated, and
the velocity and variance distributions are added as features.

### External Libraries

os for deleting old log files each run
math.floor() function for getting the bin when calculating probabilities.
numpy.square() function for variance calculation