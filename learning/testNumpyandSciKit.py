''' A script to test that numpy and scikit work properly. It takes in
two matricies and uses a neural net model to check them. A 1 is returned
if they are reasonable, and a 0 otherwise '''

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Function to test numpy is installed and works through Conda
def checkNumPy(array1, array2):
	x = np.add(array1, array2)
	return 1



# Function to test that Scikit-Learn works correctly and is installed on the user's machine
def nnModelCheck (testMat1, testMat2, out)
     #Train input
     clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
     trained = clf.fit(X, y)
     #Test
     test = clf.predict([[2., 2.], [-1., -2.]])

    #Validate
    score = 0
    for i in range(len(testMat1)):
        if test[i] == labels[i]:
            score += 1

    #Final Score
    score /= float(test.size) 
	score *= 100
	# If 75% accurate proceed
    if score >= 75:
        return 1
    else:
        return 0
        

def main()
    testMat1 = [[0., 0.], [1., 1.]]
    testMat2 = [[1., 1.], [1., 1.]]
	#Neural network output
    out = [0,1]
	print "TESTING that Conda is installed to run this Slicer Extension"
	# Test Numpy if this passes, it is installed
	checkNumPy(testMat1, testMat2)
	print "Numpy installed to Conda"
	# Test neural network output
    result = nnModelCheck(testMat1, testMat2, out)
	if result == 1:
		print "Scikit-Learn Test Passed!"
		return 1
	else:
		print "ERROR: Scikit-Learn Test Failed!"
		return 0
	


main()
