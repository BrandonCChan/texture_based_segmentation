# Texture Based Segmentation Mapping
# This is a model that performs classification based on a SciKit-Learn model serialized using Pickle
# In this case a MLP neural network is used.
# This script takes a console argument

import pickle
import numpy as np
import sys
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

# Import file data
inputFile = sys.argv[1]
data_file = open(inputFile, 'r')
data_file.readline()
data = np.loadtxt(data_file, delimiter=',')

# Read in CLF model
with open('neuralnetwork.pkl', 'rb') as f:
	clf = pickle.load(f)

# Normalize input data
RGB_data = data[:, 1:4]
RGB_data = preprocessing.normalize(RGB_data, norm='l2', axis=1)

# Classify points
results = clf.predict(RGB_data)

# Concatenate results with data
final_data = np.concatenate((data, results), axis=1)

with open('classified_texture.pkl', 'wb') as f:
	pickle.dump(clf, f)