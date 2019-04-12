
import numpy as np
import sys
import numpy as np
#from sklearn.svm import SVC
import matplotlib.pyplot as plt;
from Model.SVM import SVM

def load(file_name) :
    # X for features, y for labels
    X, y = [], [],
    tempdata = [0, 0]
    with open(file_name) as f :
        lines = f.read().splitlines()
        for line in lines[9:] :
            data = line.split()
            tempdata[0]=float(data[1])
            tempdata[1]=float(data[2][0:14])
            X.append(tempdata[0:])
            y.append(float(data[0]))


            #X.append(map(float, data[1 : ]))
        #print(data)
    return np.array(X),np.array(y)


def PlotMargin(X1_train, X2_train, clf):
    def f(x, w, b, c=0):
        # given x, return y such that [x,y] in on the line
        # w.x + b = c
        return (-w[0] * x - b + c) / w[1]

    pl.plot(X1_train[:,0], X1_train[:,1], "ro")
    pl.plot(X2_train[:,0], X2_train[:,1], "bo")
    pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

    # w.x + b = 0
    a0 = -4; a1 = f(a0, clf.w, clf.b)
    b0 = 4; b1 = f(b0, clf.w, clf.b)
    pl.plot([a0,b0], [a1,b1], "k")

    # w.x + b = 1
    a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
    b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
    pl.plot([a0,b0], [a1,b1], "k--")

    # w.x + b = -1
    a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
    b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
    pl.plot([a0,b0], [a1,b1], "k--")

    pl.axis("tight")
    pl.show()

def PlotContour(X1_train, X2_train, clf):
    pl.plot(X1_train[:,0], X1_train[:,1], "ro")
    pl.plot(X2_train[:,0], X2_train[:,1], "bo")
    pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

    X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.project(X).reshape(X1.shape)
    pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

    pl.axis("tight")
    pl.show()



if __name__ == "__main__":
    import pylab as pl

    TrainDataFile="/Data/HW1_train.rtf"
    X_train, y_train = load(TrainDataFile)

    TestDataFile="/Data/HW1_test.rtf"
    X_test, y_test = load(TestDataFile)

    print(y_test)
    clf = SVM(C=0.1)             #soft margin
    #clf=SVM(gaussian_kernel)    



    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    PlotContour(X_train[y_train==1], X_train[y_train==-1], clf)