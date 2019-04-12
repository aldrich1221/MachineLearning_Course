import numpy as np;
import matplotlib.pyplot as plt;
import random;
from Model.RidgeRegression import RidgeRegression
def My_read(file_name) :
    X, y = [], []
    

    with open("Data/"+file_name) as f :
        for line in f :
            data = line.split()           
            temp=np.zeros(len(data)-1);            
            for xx in range(len(data)-1):                
                temp[xx]=float(data[xx])
            X.append(temp)
            y.append(int(data[-1]))
    return X, y
X_train,Y_train=My_read('hw4_train.dat.txt');
X_test,Y_test=My_read('hw4_test.dat.txt');


#7->14,15
model = RidgeRegression()
lambArray=10**np.arange(-10.0,10);  
index=0;
ErrorIn=np.zeros(len(lambArray))
ErrorOut=np.zeros(len(lambArray))
for lamb in lambArray:
    
    model.fitting(X_train,Y_train, lamb)
    ErrorIn[index]=model.Error(X_train,Y_train);
    ErrorOut[index]=model.Error(X_test,Y_test);
    
    index=index+1;



#8->1617
Xval, Yval = X_train[120 : ], Y_train[120 : ]
X_train2, Y_train2 = X_train[ : 120], Y_train[ : 120]

model2 = RidgeRegression()
lambArray=10**np.arange(-10.0,10);  
index2=0;
ErrorVal=np.zeros(len(lambArray))
ErrorTrain=np.zeros(len(lambArray))
ErrorOut=np.zeros(len(lambArray))
for lamb in lambArray:
    model2.fitting(X_train2,Y_train2, lamb)
    ErrorVal[index2]=model2.Error(Xval,Yval);
    ErrorTrain[index2]=model2.Error(X_train2,Y_train2);
    ErrorOut[index2]=model2.Error(X_test,Y_test);
    print(lamb)
    print(ErrorVal[index2])
    print(ErrorTrain[index2])
    index2=index2+1;

plt.plot(ErrorVal,'g--')  
plt.plot(ErrorTrain,'r') 
plt.plot(ErrorOut,'b') 
plt.title("Error plot")
plt.show()




