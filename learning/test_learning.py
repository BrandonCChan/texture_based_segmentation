import pickle
import numpy as np

# Import file data
data_file = open('Manual_Segment_Short.csv')
data_file.readline()
data = np.loadtxt(data_file, delimiter=',')

with open('neuralnetwork.pkl', 'rb') as f:
	clf = pickle.load(f)

# Split labels and training data
data_labels = data[:, 0]
RGB_data = data[:, 1:4]

# Test model based on randomly sampled data
test_results = clf.predict(RGB_data)

# Calculate score accuracy to validate results of the model
score = 0
for i in range(len(test_results)):
	if test_results[i] == data_labels[i]:
		score += 1

# Calculate final score based on % = (# correct classifications) / (sample size)
score /= float(test_results.size)

# Convert to percentage
score *= 100

print score