''' A script to test that numpy and scikit work properly. It takes in
two matricies and uses a neural net model to check them. A 1 is returned
if they are reasonable, and a 0 otherwise '''

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


#Input Matrix
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
        score /= flaot(test.size score *= 100

        if score is >= 75
                return 1
        else
                return 0
        

def main()
        testMat1 = [[0., 0.], [1., 1.]]
        testMat2 = 
        out = [0,1]

        result = nnModelCheck(testMat1, testMat2, out)
        print result


main()
