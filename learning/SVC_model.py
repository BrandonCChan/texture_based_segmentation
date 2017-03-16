# Alternative model for texture based classification
# Attempt to classify tissue based on RGB values using SVC - C support vector classification
# This is done using Conda and Scikit-learn aimed to be compatible with 3D Slicer
# Heavily based on the tutorials found on Scikit.org
#
# Nuwan Perera, Brandon Chan, Mareena Mallory

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# SVC parameter variables
h = 0.02 # Mesh step size
C = 1.0 # SVM regularization

# Import data file
data_file = open('Manual_Segment_Short.csv')
data_file.readline()
data = np.loadtxt(data_file, delimiter=',')

# Split labels and training data
data_labels = data[:, 0]
RGB_data = data[:, 1:4]

# Normalize RGB data to between 0 and 1
RGB_data = preprocessing.normalize(RGB_data, norm='l2')

# Randomly sample dataset
input_data, test_data, data_labels, test_labels = train_test_split(RGB_data, data_labels, test_size=0.4, random_state=0)

# Run SVC
svc = svm.SVC(kernel='linear', C=C).fit(input_data, data_labels)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(input_data, data_labels)
poly_svc = svm.SVC(kernel='poly',degree=3, C=C).fix(input_data, data_labels)
lin_svc = svm.LinearSVC(C=C).fit(input_data, data_labels)

# Create a mesh plot
x_min, x_max = input_data[:,0].min() - 1, input_data[:,0].max() + 1
y_min, y_max = input_data[:,1].min() - 1, input_data[:,1].max() + 1
xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# title the plots
titles = ['SVC with linear kernel' , 'LinearSVC', 'SVC with RBF', 'SVC with polynomial']

for i,clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
	# Plot the decision boundary
	# point in the mesh
	plt.subplot(2,2,i+1)
	plt.subplots_adjust(wspace=0.4, hspace=0.4)
	
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	
    # Put the result into a color plot
	# BUG: indentation problem
    Z = Z.reshape(xx.shape)
	
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()

