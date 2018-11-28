#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 18:34:25 2018

@author: cdz
"""
import numpy as np
import matplotlib.pyplot as plt
import operator
import os
import cv2

#########################################################2.1
def createDataSet():
    group=np.array([[1.0,1.1],
                 [1.0,1.0],
                 [0,0],
                 [0,0.1]])
    labels=['A','A','B','B']
    return group,labels


def classify0(inX,dataSet,labels,k):
    '''
    inX: the sample to be classified
    dataSet: the data with labels
    labels:,k:.....
    '''
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**2
    sortedDistIndicies=distances.argsort()  # get the index of the sorted data
    classCount={}
    for i in range(k):
        voteILabel=labels[sortedDistIndicies[i]]  # label of the ith nearest sample
        classCount[voteILabel]=classCount.get(voteILabel,0)+1 # (key:value); if not in classCount,get will return 0
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

        
######################################################  2.2

def file2matrix(filename):
    f=open(filename)
    arrayOLines=f.readlines()
    numberOLines=len(arrayOLines)
    returnMat=np.zeros((numberOLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()  # split Enter kry
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
        
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVal=dataSet.min(0)
    maxVal=dataSet.max(0)
    ranges=maxVal-minVal
    normDataSet=np.zeros(dataSet.shape)
    m=dataSet.shape[0]
    normDataSet=dataSet-np.tile(minVal,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVal
        
    ''' # I have no test file, so I didn't write this
def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    normMat,ranges,minVal=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(mormMat[i,:])
    '''
    
def classifyPerson():
    resultList=['not all','in small does','in large does']
    ffMiles=float(input('frequent flier miles earned per year?'))
    percentTats=float(input('percentage of time spent playing video games?'))
    iceCream=float(input('liters of ice cream consumed per year?'))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVal=autoNorm(datingDataMat)
    inArr=np.array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVal)/ranges,normMat,datingLabels,3)
    print('you will probably like this person:',\
          resultList[classifierResult-1])

###################################################### 2.3
def img2vector(filename):
    returnVec=np.zeros((1,1024))
    f=open(filename)
    for i in range(32):
        linStr=f.readline()
        for j in range(32):
            returnVec[0,32*i+j]=int(linStr[j])
    return returnVec


def handwritingClassTest(testPicVex):
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    
    result=classify0(testPicVec,trainingMat,hwLabels,3)
    print('we predict that your number is:',result)
    # the following is used to test the accuracy, acutually I know it is about 98% in test dataset
    '''
    testFileList = os.listdir('digits/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))
   '''
   
   
   
# this function is written by me , we can use it to change picture to txt
# attention: you had better cut you picture probably (access to 32x32)
# the threshold is set by watching the value of the picture pixels(you had better use ostu)
def img2txt(imgPath):
    pic=cv2.imread(imgPath,0)
    f=open('testPic.txt','w')
    size=(32,32)
    resizePic=cv2.resize(pic,size)
    shape=resizePic.shape
    for i in range(shape[0]): #height
        for j in range(shape[1]):
            if resizePic[i,j]<100:
                f.write('1')
            else:
                f.write('0')
        f.write('\n')
        
    
    




#classifyPerson() 
img2txt('testPic.png')
testPicVec=img2vector('testPic.txt')
handwritingClassTest(testPicVec)  



    
    
    
    
    
   
#test
'''
group,labels=createDataSet()
print(group)
print(labels) 

x=group[:,0]
y=group[:,1]
color=['r','g']
plt.scatter(x,y,c=color)
'''
'''
x=np.array([[1,2],
            [3,5],
            [6,3],
            [9,4]])
y=np.array([2,5])
z=x-y
print(x)
print(x**2)
'''




























