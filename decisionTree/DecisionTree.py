#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:39:01 2018

@author: cdz
"""
import math
import operator


# used for test
def createDataSet():
    dataSet=[
            [1,1,'yes'],
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']]
    labels=['no surfacing', 'flippers']
    return dataSet,labels

# caculate entropy
def getEntropy(dataSet):
    '''
    input:dataSet with labels
    return: H of the dataSet
    '''
    num=len(dataSet)
    labelCounts={}
    
    # get numbers of each class
    for featureVec in dataSet:
        currLabel=featureVec[-1] # label of this sample
        if currLabel not in labelCounts.keys():
            labelCounts[currLabel]=0
        labelCounts[currLabel]+=1
    
    # caculatre entropy
    H=0.0 # H is entropy
    for key in labelCounts:
        p=float(labelCounts[key])/num
        H-=p*math.log(p,2)
    return H
    
# split dataset, to fully understand this function, just look at bestFeatToSplit
def splitDataSet(dataSet,axis,value):
    '''
    inout:
        dataSet:
        axis: the index of the feature used to split
        value: ....
    return:
        splitedDataSet: the dataSet after split the samples which have value in feature[axis]
    '''
    splitedDataSet=[]
    for featureVec in dataSet:
        if featureVec[axis]==value:
            reducedFeatureVec=featureVec[:axis]    ##why split attribute axis  ????
            reducedFeatureVec.extend(featureVec[axis+1:])
            splitedDataSet.append(reducedFeatureVec)
    return splitedDataSet


# get the best feature to split
def bestFeatToSplit(dataSet):
    nFeatures=len(dataSet[0])-1
    bestEntropy=getEntropy(dataSet)
    bestInfoGain=0.0
    bestFeature=-1 # index
    
    # look at  https://blog.csdn.net/u012351768/article/details/73469813
    for i in range(nFeatures): # for each feature
        featureList=[example[i] for example in dataSet]
        #featureList=dataSet[:,i] # all data of samples in feature[i]
        uniqueVals=set(featureList) #set!!!!!, use each value to split dataSet
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            p=len(subDataSet)/float(len(dataSet))
            newEntropy+=p*getEntropy(subDataSet)
        infoGain=bestEntropy-newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    
    return bestFeature
         
    
# vote
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.inemgetter(1),reverse=True)
    return sortedClassCount[0][0]


# recursion to create tree
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    
    # if the class of each sample in the dataset is the same, we return
    if classList.count(classList[0])==len(classList):
        return classList[0]
    
    # if had gone through all attribution, choose the most class to return
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    
    bestFeature=bestFeatToSplit(dataSet)
    bestFeatLabel=labels[bestFeature]
    
    # in each node, use the best feature to create son node
    myTree={bestFeatLabel:{}}
    del(labels[bestFeature])  # delete the best attribute
    featureVals=[example[bestFeature] for example in dataSet]
    uniqueVals=set(featureVals)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeature,value),subLabels)
    
    return myTree

        
    
# calssfication
def classify(inputTree,featureLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featureIndex=featureLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featureIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featureLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel
                
    
# store the decison tree
def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)

    

    

    
    
    
    


if __name__=='__main__':
    '''
    myDat,labels=createDataSet()  
    label=labels.copy() # labels will change in createTree
    myTree=createTree(myDat,labels)
    storeTree(myTree,'decisionTree.txt')
    
    readTree=grabTree('decisionTree.txt')
    clas=classify(readTree,label,[1,1])
    print(clas)
    '''
    
    fr=open('lenses.txt')
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    lenseLabels=['age','prescript','astigmatic','tearRate']
    lenseLabel=lenseLabels.copy()
    lenseTree=createTree(lenses,lenseLabels)
    storeTree(lenseTree,'decisionTree.txt')
    readTree=grabTree('decisionTree.txt')
    
    testVec=['young','hyper','no','normal','soft']
    clas=classify(readTree,lenseLabel,testVec)
    print(clas)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
