

import random
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt;
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import math


def load(file_name) :
    # X for features, y for labels
    X, y ,Xtemp= [[0,0]],[[0]],[]
    #tempdata = [[], 0]
    with open(file_name) as f :
       
        lines = f.read().splitlines()
        for line in lines :
            data = line.split()
            Xtemp=[float(data[0]),float(data[1])]
            X=np.append(X,[Xtemp], axis=0)
            y=np.append(y,[[float(data[2])]], axis=0)
            #X.append(map(float, data[1 : ]))

    return np.array(X[1:]),np.array(y[1:])


def buildStump(dataArr,classLabels,D,testx,testy,DD):
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    testxMatrix=np.mat(testx);testyMat=np.mat(testy).T
    
    m,n = np.shape(dataMatrix)
    mm,nn=np.shape(testxMatrix)
   
    numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
    bestTestClasEst = np.mat(np.zeros((mm,1)))
   
    minError = math.inf 
    TestminError=math.inf
   
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
       
        stepSize = (rangeMax - rangeMin) / numSteps
       
        for j in range(-1, int(numSteps) + 1):
            
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                

                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                
                TestpredictedVals = stumpClassify(testxMatrix,i,threshVal,inequal)

                errArr = np.ones((m,1))
                testerrArr=np.ones((mm,1))
                
                errArr[(predictedVals == labelMat)] = 0 
                testerrArr[(TestpredictedVals == testyMat)] = 0 
                
                
                weightedError = D.T*errArr
                TestweightedError=DD.T*testerrArr
                #print(weightedError)
                
                #print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError[0]))
                if weightedError < minError:
                  
                    minError = weightedError
                   
                    A=np.sum(errArr)/len(errArr)
                    
                    TestminError = TestweightedError
                    bestClasEst = predictedVals.copy()
                    bestTestClasEst = TestpredictedVals.copy()
                    bestStump['dim'] = i 
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    
    return bestStump,minError,bestClasEst,TestminError,bestTestClasEst,A


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    

    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
        
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray



def adaBoostTrainDS(dataArr,classLabels,testx,testy,numIt=40):
    weakClassArr = []
   
    
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)
    
    mm = np.shape(testx)[0]
    DD = np.mat(np.ones((mm,1))/mm)
    
    
    
    aggClassEst = np.mat(np.zeros((m,1)))
    testaggClassEst = np.mat(np.zeros((mm,1)))
    
    
    
    G_Error=np.zeros(numIt)
    g_Error=np.zeros(numIt)
    
    Test_G_Error=np.zeros(numIt)
    Test_g_Error=np.zeros(numIt)
    U_t=np.zeros(numIt)
    
    
    for i in range(numIt):
        bestStump,error,classEst,testerror,testclassEst,A = buildStump(dataArr,classLabels,D,testx,testy,DD)
       
        
        alpha = float(0.5 * math.log((1.0 - error) / max(error,1e-16)))
        if i ==0:
            alpha1=alpha
            error1=error
        
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
       
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        
        Temp=np.zeros((m,1))
        for temp in range(len(expon)):
            Temp[temp]=math.exp(expon[temp])
       
        D = np.multiply(D[:], Temp)
        
        U_t[i]=np.sum(Temp)/m
        
        D = D / D.sum()
        
        aggClassEst += alpha*classEst
        testaggClassEst += alpha*testclassEst
        
        
        
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        testaggErrors = np.multiply(np.sign(testaggClassEst) != np.mat(testy).T,np.ones((mm,1)))
        
       
        errorRate = aggErrors.sum()/m
        testerrorRate=testaggErrors.sum()/mm
        
        G_Error[i]=errorRate
        g_Error[i]=A
        Test_G_Error[i]=testerrorRate
        Test_g_Error[i]=testerror
        
        
        if errorRate == 0.0: break
        
    return weakClassArr,errorRate,G_Error,g_Error,alpha1,error1,U_t,Test_G_Error,Test_g_Error
def loadSimpData():
    datMat = np.matrix([[ 1. , 2.1],
        [ 2. , 1.1],
        [ 1.3, 1. ],
        [ 1. , 1. ],
        [ 2. , 1. ]]) 

    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels


LABEL = 2
U_WEIGHT = 3
DIR_POS = 1
DIR_NEG = -1
DataFile_train="Data/hw3_train.txt"
DataFile_test="Data/hw3_test.txt"
    
X_train, y_train= load(DataFile_train)
X_test, y_test =  load(DataFile_test)
T =100
N=len(y_train)    #data length
D=2                 #X dimension
u = np.ones(len(X_train)) / len(X_train)
data_train =np.append(X_train,y_train.reshape(-1, 1),axis=1)
data_train=np.append(data_train,u.reshape(-1, 1),axis=1)

#tt, ll = loadSimpData()
t=X_train
l=y_train.reshape(100)
testx=X_test
testy=y_test.reshape(1000)



m = np.shape(t)[0]
mm=np.shape(testx)[0]

D = np.mat(np.ones((m,1))/m)
DD=np.mat(np.ones((mm,1))/mm)
ErrorArray=np.zeros(299)
errorArray=np.zeros(301)

classifierArray,GT_error,Gin_error,gin_error,alpha1,error1,U_t,Test_G_Error,Test_g_Error = adaBoostTrainDS(t, l,testx,testy, 1000)
#print(classifierArray)
plt.plot(Gin_error,label='U_t')
plt.title('U_T vs t')

plt.legend()

plt.show()



