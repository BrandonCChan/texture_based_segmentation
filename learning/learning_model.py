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
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Import file data
data_file = open('Manual_Segment_Short.csv')
data_file.readline()
data = np.loadtxt(data_file, delimiter=',')

# Split labels and training data
data_labels = data[:, 0]
RGB_data = data[:, 1:4]

# Normalize RGB data to between 0 and 1
RGB_data = preprocessing.normalize(RGB_data, norm='l2')

# Randomly sample dataset
train_data, test_data, train_labels, test_labels = train_test_split(RGB_data, data_labels, test_size=0.4, random_state=0)


# Instantiate MLP with quasi-Netownian optimization methods, 3 layers - 9 nodes per layer and train model
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(9,9,9), random_state=1)
clf.fit(train_data, train_labels)

# Test model based on randomly sampled data
test_results = clf.predict(test_data)

# Calculate score accuracy to validate
score = 0
for i in range(len(test_data)):
	if test_results[i] == test_labels[i]:
		score += 1

# Calculate final score
score /= len(test_data)

print score
