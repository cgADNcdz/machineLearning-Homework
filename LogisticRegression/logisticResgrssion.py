#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:16:12 2018

@author: cdz
"""

import numpy as np
import matplotlib.pyplot as plt



# get test dataset
def loadDataSet(file):
    data=[]
    labels=[]
    fr=open(file,'r')
    datas=fr.readlines()
    for line in datas:
        lineList=line.split()
        data.append([1.0,float(lineList[0]),float(lineList[1])]) # x0=1, Corresponding to bias(b)
        labels.append(float(lineList[2]))
    fr.close()      
    return data,labels
    
#sigmoid function
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# grad ascent
def gradAscent(data,labels):
    dataMat=np.mat(data)
    labelMat=np.mat(labels).transpose()
    m,n=np.shape(dataMat)
    alpha=0.001  #learning rate
    nStep=500    #number of most iterations
    weights=np.ones((n,1))  # w (n,1)
    for i in range(nStep):
        '''
        to understand the following, look at:https://www.cnblogs.com/zy230530/p/6875145.html
        Actually, i just caculate aL(w)/wj to understand!
        '''
        h=sigmoid(dataMat*weights)  #(m,1)  h=f(w)
        error=labelMat-h            #(m,1)  
        weights=weights+alpha*dataMat.transpose()*error
    return weights


# plot
def plotBestFit(weights):
    data,label=loadDataSet('testSet.txt')
    dataArray=np.array(data)
    n=np.shape(data)[0]
    xcoord1=[];xcoord2=[]
    ycoord1=[];ycoord2=[]
    for i in range(n):
        if int(label[i])==1:
            xcoord1.append(dataArray[i,1]);ycoord1.append(dataArray[i,2])
        else:
            xcoord2.append(dataArray[i,1]);ycoord2.append(dataArray[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcoord1,ycoord1,s=30,c='red',marker='s')
    ax.scatter(xcoord2,ycoord2,s=30,c='green')
    x=np.mat(np.arange(-3.0,3.0,0.1))
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('x1'),plt.ylabel('x2')
    plt.show()    
    
    
def stocGradAscent0(data,label):
    '''
    in tihs function, a batch is just a sample(actually we can use not only a sample)
    '''
    m,n=np.shape(data)
    alpha=0.1
    weights=np.ones(n)
    for i in range(m):
        h=sigmoid(sum(data[i]*weights))
        error=label[i]-h
        weights=weights+alpha*error*data[i]
    return weights




'''
def stocGradAscent1(data,label,nStep=150):
   
    #in this function, alpha can change smaller when tanining
    #and each sample we choose randomly
    
    m,n=np.shape(data)
    weights=np.ones(n)
    for j in range(nStep):
        dataIndex=range(m)
        for i in range(m):
            alpha=4/(1.0+i+j)+0.01
            randIndex=int(np.random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(data[randIndex]*weights))
            error=label[randIndex]-h
            weights=weights+alpha*error*data[randIndex]
            del(dataIndex[randIndex])
    return weights
  '''
  
  
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(np.random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights          
            
        
def classifyVector(x,weights):
    prob=sigmoid(sum(x*weights))
    if prob>0/5:
        return 1
    else:
        return 0.0
    

def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    
    # training
    trainingSet=[]
    trainingLabel=[]
    for line in frTrain.readlines():  # read data
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabel.append(float(currLine[21]))  
    trainWeights=stocGradAscent1(np.array(trainingSet),trainingLabel,500) #traiing
    
    #test
    errorCount=0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print("the errorRate of this test is:",errorRate)
    return errorRate

def multiTest():
    numTests=10;errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print('after',numTests,'iterations,the average error rate is:',errorSum/numTests)
    
    
        
        





####### test 
if __name__=='__main__':
    data,label=loadDataSet('testSet.txt')
    #print(label)
    #weights=gradAscent(data,label)
    #print(weights)
    #plotBestFit(weights)
    #weights=gradAscent(data,label)
    #plotBestFit(weights)
    #weights=gradAscent(data,label)
    #plotBestFit(weights)
    multiTest()
    print('\n   i debug many times but i can not understand the error rate should be the same,does the randInex not work?')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    