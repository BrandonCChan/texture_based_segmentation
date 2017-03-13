# Implementation of machine learning for classification of tissue for texture based segmentation
# This uses Conda for Windows and Sci-Kit learn for building models

# With a 4-class classification problem based on 3 attributes, aim to use a 3 layered perceptron to predict tissue type.
# This model will train using backpropagation.
# The final model will need to be unsupervised since data is not labelled, rather it aims to make a prediction without
# knowing the correct answer.

import numpy as np
from sklearn.neural_network import MLPClassifier

# Normalize RGB data to a Gaussian curve
data_normalized = preprocessing.normalize(data, norm='l2')

# Instantiate MLP with stochastic gradient descent, 3 layers - 9 nodes per layer
clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes(9,9,9), random_state=1)

# This will need training data to build a proper model
train_data = []
train_labels = []

clf.fit(train_data, train_labels)

# Predict based on test data
test_data = []
test_labels = []

# I think the results of this prediction is a numpy array - need to confirm
test_results = clf.predict(test_data)

score = 0
for i in range(len(test_data)):
	if test_results[i] == test_labels[i]:
		score += 1

# Calculate final score
score /= len(test_data)
