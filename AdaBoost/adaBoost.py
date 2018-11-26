#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 20:35:03 2018

@author: cdz
"""
import numpy as np 
import math

def loadSimpleData():
    dataMat=np.matrix([
            [1.0,2.1],
            [2.0,1.1],
            [1.3,1.0],
            [1.0,1.0],
            [2.0,1.0]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels


def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split("\t")
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat








# test if > or <  threshOld
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray=np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq=="lt":
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray


# find best one layer decision boundary
def buildStump(dataArray,classLabels,D):
    dataMatrix=np.mat(dataArray)
    labelMat=np.mat(classLabels).T
    m,n=np.shape(dataMatrix)
    numSteps=10.0
    bestStump={}
    bestClassEst=np.mat(np.zeros((m,1)))
    minError=np.inf
    for i in range(n):  # for each feature
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps   #length of step
        for j in range(-1,int(numSteps)+1):  # for each step
            for inequal in ["lt","gt"]:
                threshVal=(rangeMin+float(j)*stepSize)
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr=np.mat(np.ones((m,1)))
                errArr[predictedVals==labelMat]=0
                weightedError=D.T*errArr     # cala error with weight
                #print("split:dim ",i," thresh ",threshVal," thresh inequal: ",inequal," the wieghted error is ",weightedError)                                    
                if weightedError<minError:
                    minError=weightedError
                    bestClassEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    
    return bestStump,minError,bestClassEst



# DS means decision tree with only one layer
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr=[]
    m=np.shape(dataArr)[0]
    D=np.mat(np.ones((m,1))/m)
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        #print("D:",D.T)
        alpha=float(0.5*math.log((1.0-error)/max(error,1e-16)))
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        #print("classEst: ",classEst.T)
        expon=np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
        D=np.multiply(D,np.exp(expon))
        D=D/D.sum()
        aggClassEst+=alpha*classEst
        #print("aggClassEst: ",aggClassEst.T)
        aggErrors=np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))
        errorRate=aggErrors.sum()/m
        print("totla error: ",errorRate)
        if errorRate==0.0:
            break
    return weakClassArr



def adaClassify(dataToClass,classifierArr):
    dataMatrix=np.mat(dataToClass)
    m=np.shape(dataMatrix)[0]
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
       # print(aggClassEst)
    return np.sign(aggClassEst)


    






if __name__=="__main__":
    '''
    dataMat,classLabels=loadSimpleData()
    #D=np.mat(np.ones((5,1))/5)
   
    bestStump,minError,bestClassEst=buildStump(dataMat,classLabels,D)
    print(bestStump)
    print(minError)
    print(bestClassEst)
  
    weakClassArr=adaBoostTrainDS(dataMat,classLabels,9)
    clas=adaClassify([0,0],weakClassArr)
    print(clas)
    '''
    
    dataArr,labelArr=loadDataSet("horseColicTraining2.txt")
    print("error on training")
    classifierArray=adaBoostTrainDS(dataArr,labelArr,40)
    
    testArr,testLabelArr=loadDataSet("horseColicTest2.txt")
    predicted40=adaClassify(testArr,classifierArray)
    print("predict classes on test data")
    print(predicted40)
    
    print("test error on test data")
    testError=(np.sum(np.abs(np.mat(testLabelArr).T-predicted40))/2)/np.shape(predicted40)[0]
    print(testError)
    
    
    
    
    
    
    
    
    
    
    