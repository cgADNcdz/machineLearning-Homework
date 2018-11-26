#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:13:35 2018

@author: cdz
"""
import random
import numpy as np 


def loadDataSet(fileName):
    dataMat=[];labelMat=[]
    f=open(fileName)
    lines=f.readlines()
    for line in lines:
        lineArray=line.strip().split('\t')
        dataMat.append([float(lineArray[0]),float(lineArray[1])])
        labelMat.append(float(lineArray[2]))
    return dataMat,labelMat
   
    
# select j randomly
def selectJ(i,m):
    '''
    i is the subscript of a, m is the number of alpha
    '''
    j=i
    while(j==i):
        j=int(random.uniform(0,m)) # random number in 0 to m
    return j


#
def clipAlpha(aj,H,L):
    '''
    H: higth threshold of a2
    L:low threshlod of a2
    look at lihang's book, p127
    '''
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj
        



# a simple version of smo
def smoSimple(dataMat,classLabel,C,toler,numIter):
    dataMat=np.mat(dataMat); labelMat=np.mat(classLabel).transpose()
    b=0
    m,n=np.shape(dataMat)
    alphas=np.mat(np.zeros((m,1)))
    iters=0
    while(iters<numIter):
        alphaPairsChanged=0  # if a1,a2 have changed
        for i in range(m):
            '''
            here we not use kernal. for more deltais, look at p127 
            '''
            fXi=float(np.multiply(alphas,labelMat).T*(dataMat*dataMat[i,:].T))+b # fXi is the predicted result in data[i]
            Ei=fXi-float(labelMat[i])
            if ((labelMat[i]*Ei)<-toler and (alphas[i]<C)) or ((labelMat[i]*Ei>toler)and (alphas[i]>0)):
                '''
                choose first a
                '''
                j=selectJ(i,m)
                
                fXj=float(np.multiply(alphas,labelMat).T*(dataMat*dataMat[j,:].T))+b
                Ej=fXj-float(labelMat[j])
                
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                
                if(labelMat[i]!=labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H:
                    print('L=H')
                    continue
                eta=2.0*dataMat[i,:]*dataMat[j,:].T-dataMat[i,:]*dataMat[i,:].T-dataMat[j,:]*dataMat[j,:].T
                if eta>=0:
                    print('eta>=0')
                    continue
                
                alphas[i]-=labelMat[i]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)
                if(abs(alphas[j]-alphaJold)<0.00001):
                    print("j not moving enough")
                    continue
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMat[i,:]*dataMat[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMat[i,:]*dataMat[j,:].T       
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMat[i,:]*dataMat[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMat[j,:]*dataMat[j,:].T
                if((0<alphas[i])and (C>alphas[i])):
                    b=b1
                elif((0<alphas[j]) and (C>alphas[j])):
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaPairsChanged+=1
                print('iter:',iters,',i:',i,'changed:',alphaPairsChanged)
        if(alphaPairsChanged==0):
            iters+=1
        else:
            iters=0
        print("iteration number:",iters)
    return b,alphas








if __name__=="__main__":
    dataArr,labelArr=loadDataSet('testSet.txt')
    #print(labelArr)
    b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    