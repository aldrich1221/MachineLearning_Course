

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
    # numpy的mat可以把python的list轉成數學上的matrix格式
    # dataMatrix為training set, labelMat為training set的class label
    m,n = np.shape(dataMatrix)
    mm,nn=np.shape(testxMatrix)
    # shape是算出matrix維度, m在這邊實際上就是筆數, n就是feature數量
    numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
    bestTestClasEst = np.mat(np.zeros((mm,1)))
    # numSteps就是上面所提的, 切成10等份共11份來找
    # numpy.zeros(m, n) 會組成一個零矩陣 m x n
    minError = math.inf 
    TestminError=math.inf
    # 最後會選錯誤率最低的Stump, 這邊先設定成最大, numpy有inf變數意思為無窮大的值
    # 所以就開始一個一個feature開始跑
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        # 找出此feature的最大值ˊ, 最小值
        stepSize = (rangeMax - rangeMin) / numSteps
        # 根據前面設定的步數設定出等一下要切的數量
        for j in range(-1, int(numSteps) + 1):
            # 那判斷有兩個方向, 一個是比大, 一個比小, 這邊他有把邊界最小直納入考量, 最大反而沒有
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                # 這邊只是計算一下現在是判斷到哪一個切割點

                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                # 這就是把training set, 哪一個feature, 切割點, 比大還是比小傳入stumpClassify
                
                TestpredictedVals = stumpClassify(testxMatrix,i,threshVal,inequal)

                errArr = np.ones((m,1))
                testerrArr=np.ones((mm,1))
                #設定錯誤的vector, 先假設都1, 都錯的初始
               
                #print((predictedVals == labelMat))
                #print(errArr)
                #errArr[(predictedVals == labelMat)[0]] = 0 
                errArr[(predictedVals == labelMat)] = 0 
                testerrArr[(TestpredictedVals == testyMat)] = 0 
                
                # 有猜對的話, 就改成為0
                #print(errArr)
                #print(D)
                weightedError = D.T*errArr
                TestweightedError=DD.T*testerrArr
                #print(weightedError)
                # 算錯誤率, 要乘上weight
                #print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError[0]))
                if weightedError < minError:
                    # 找哪個是最好的Decision Stump切割
                    
                    minError = weightedError
                   
                    A=np.sum(errArr)/len(errArr)
                    
                    TestminError = TestweightedError
                    bestClasEst = predictedVals.copy()
                    bestTestClasEst = TestpredictedVals.copy()
                    bestStump['dim'] = i 
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    
    return bestStump,minError,bestClasEst,TestminError,bestTestClasEst,A

# 這個function判斷完切割點之後, 就會猜測這是1 or -1, 也就是class label, 然後會把猜測的回傳
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    # 先設定猜測每筆都是1, 只是設定初值而已, 別太在意

    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
        # 這個就只是判斷值有沒有小於等於切割點, 有的話就是-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray



def adaBoostTrainDS(dataArr,classLabels,testx,testy,numIt=40):
    weakClassArr = []
   
    # 用來存放等會每回合會產生的weak learner
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
        # 呼叫weak learner, classEst就是ht(xi
        #print( "D:",D.T)
        
        alpha = float(0.5 * math.log((1.0 - error) / max(error,1e-16)))
        if i ==0:
            alpha1=alpha
            error1=error
        # alpha 的公式
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # 將學好的存起來
        #print ("classEst: ",classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        # 不要被這串嚇到了, 就只是 -α * y * h(x)
        
        Temp=np.zeros((m,1))
        for temp in range(len(expon)):
            Temp[temp]=math.exp(expon[temp])
       
        D = np.multiply(D[:], Temp)
        # 然後跟原本的distribution相乘, 也就是D * -α * y * h(x)
        U_t[i]=np.sum(Temp)/m
        
        D = D / D.sum()
        # 除掉Zt, 就完成了更新
        
        aggClassEst += alpha*classEst
        testaggClassEst += alpha*testclassEst
        
        
        # 把已經學過得weak learner合體
        #print ("aggClassEst: ",aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        testaggErrors = np.multiply(np.sign(testaggClassEst) != np.mat(testy).T,np.ones((mm,1)))
        
        # 合體後的learner的結果

        errorRate = aggErrors.sum()/m
        testerrorRate=testaggErrors.sum()/mm
        #print( "total error: ",errorRate,"\n")
        G_Error[i]=errorRate
        g_Error[i]=A
        Test_G_Error[i]=testerrorRate
        Test_g_Error[i]=testerror
        
        
        if errorRate == 0.0: break
        # 如果training error 0就可以停了, 然後傳出classifier
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



