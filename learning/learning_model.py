# Implementation of neural network machine learning for classification of tissue segmentation based on texture
# This uses Conda, Sci-Kit Learn for building models, and Numpy
# Please make sure these dependencies are installed to run the code.
#
# This is a test function to ensure that everything is installed properly.

# With a 4-class classification problem based on 3 attributes, aim to use a 3 layered perceptron to predict tissue type.
# This model will train using backpropagation.
# The final model will need to be unsupervised since data is not labelled, rather it aims to make a prediction without
# knowing the correct answer.

import numpy as np
from sklearn.neural_network import MLPClassifier

# Import file data
data_file = open('Manual_Segment_Short.csv')
data_file.readline()
data = np.loadtxt(f, delimiter=',')

# Shuffle the dataset and randomly sample between training and testing
# Need to find numpy or scikit documentation for this


# Split labels and training data
train_labels, data = np.hsplit(data, 2)

# Normalize RGB data to between 0 and 1
train_data = preprocessing.normalize(data, norm='l2')

# Instantiate MLP with quasi-Netownian optimization methods, 3 layers - 9 nodes per layer and train model
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(9,9,9), random_state=1)
clf.fit(train_data, labels)

'''
# Test model based on randomly sampled data
clf.predict(test_data)

Calculate score accuracy to validate
score = 0
for i in range(len(test_data)):
	if test_results[i] == test_labels[i]:
		score += 1

# Calculate final score
score /= len(test_data)
'''