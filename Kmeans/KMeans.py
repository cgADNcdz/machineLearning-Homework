#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 21:25:09 2018

@author: cdz
"""

import numpy as np 
import math
import urllib
import json
import matplotlib
import matplotlib.pyplot as plt



# load data from file
def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split("\t")
        fltLine=[float(x) for x in curLine]
        dataMat.append(fltLine)
        
    return dataMat


# caculate Euclidean distance 
def distEclud(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))

# choose ceters randomly
def randCenter(dataSet,k):
    n=np.shape(dataSet)[1]
    centerRoids=np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangeJ=float(max(dataSet[:,j]-minJ))
        centerRoids[:,j]=minJ+rangeJ*np.random.rand(k,1)
    return centerRoids


# the K Means algorithm
def KMeans(dataSet,k,distMeans=distEclud,createCenter=randCenter):
    m=np.shape(dataSet)[0]
    clusterAssment=np.mat(np.zeros((m,2)))
    centerRoids=createCenter(dataSet,k)  # choose centers randomly
    clusterChanged=True # use to decide if continue or not
    while clusterChanged:
        clusterChanged=False
        for i in range(m):   # for each sample
            minDist=math.inf
            minIndex=-1
            for j in range(k): # the distance of the ith sample and jth center
                distJI=distMeans(centerRoids[j,:],dataSet[i,:])
                if distJI<minDist:
                    minDist=distJI
                    minIndex=j
            if clusterAssment[i,0]!=minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        #print(centerRoids)
        
        for cent in range(k):  # refresh the centers
            ptsInClust=dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            centerRoids[cent,:]=np.mean(ptsInClust,axis=0)
    return centerRoids,clusterAssment


# bisecting KMeans
def biKMeans(dataSet,k,distMeans=distEclud):
    m=np.shape(dataSet)[0]
    clusterAssment=np.mat(np.zeros((m,2)))
    centRoid0=np.mean(dataSet,axis=0).tolist()[0]
    centList=[centRoid0]
    for j in range(m):
        clusterAssment[j,1]=distMeans(np.mat(centRoid0),dataSet[j,:])**2
    while(len(centList)<k):
        lowestSSE=math.inf
        for i in range(len(centList)):
            ptsInCurrCluster=dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]
            centRoidMat,splitClustAss=KMeans(ptsInCurrCluster,2,distMeans)
            sseSplit=sum(splitClustAss[:,1])
            sseNotSplit=sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit,and notSplit:",sseSplit,sseNotSplit)
            if sseSplit+sseNotSplit < lowestSSE:
                bestCentToSplit=i
                bestNewCents=centRoidMat
                bestClustAss=splitClustAss.copy()
                lowestSSE=sseSplit+sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:,0].A==1)[0],0]=len(centList)
        bestClustAss[np.nonzero(bestClustAss[:,0].A==0)[0],0]=bestCentToSplit
        print("the bestCentToSplit is: ",bestCentToSplit)
        print("the len of bestClustAss is: ",len(bestClustAss))
        centList[bestCentToSplit]=bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[np.nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:]=bestClustAss
    return centList,clusterAssment

     
'''           
### get coordinate of the point 
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print(yahooApi)
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())
'''       

# get distance of two points in the earth
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = math.sin(vecA[0,1]*np.pi/180) * math.sin(vecB[0,1]*np.pi/180)
    b = math.cos(vecA[0,1]*np.pi/180) * math.cos(vecB[0,1]*np.pi/180) * \
                      math.cos(np.pi * (vecB[0,0]-vecA[0,0]) /180)
    return np.arccos(a + b)*6371.0 #pi is imported with numpy
            
    
# clustering the plcaes    
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = biKMeans(datMat, numClust, distMeans=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    #ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()






if __name__=="__main__":
    #data=np.mat(loadDataSet("testSet.txt"))
    #centers=randCenter(data,2)
    #print(type(distEclud(data[0],data[3])))
    #centers,clustAssing=KMeans(data,4)
    #print("the centers:\n",centers)
    #print(clustAssing)
    #dataMat=np.mat(loadDataSet("testSet2.txt"))
    #centList,myNewAssment=biKMeans(dataMat,3)
    #print(centList)
    clusterClubs(5)








































