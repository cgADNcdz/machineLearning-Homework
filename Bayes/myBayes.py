#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:39:49 2018

@author: cdz
"""


class NaiveBayes:
    def __init__(self,data,label):
        '''
        input:data with the shape of (m,n)
              label with the shape of m
        input type: list
        '''
        self.data=data
        self.label=label
        self.PriorProb=[]
        self.ConditProb=[]
        self.num=len(self.label)
        
  
      
    def caculProb(self):
        classes=list(set(self.label)) # caculate how many classes in labels
        K=len(classes)       # number of classes
        #num=len(self.label) #number of training data
        
        # caculate proirProbility
        for i in range(K):
            self.PriorProb.append(0.0)   # avoid index out of range
            for j in range(self.num):
                if self.label[j]==classes[i]:
                    self.PriorProb[i]+=1
            self.PriorProb[i]/=self.num
           # print(self.PriorProb[i])
        
        # caculate conditionProbility
        # the main loop: i j l
        dim=len(self.data[0])  # dimension of each data
        for i in range(K):     # caculate condition Probility in each class
            self.ConditProb.append([])
            for j in range(dim):   #P(Xj=al|Y=Ck)
                colj=[]
                for x in range(self.num):
                    try:
                        colj.append(self.data[x][j])
                    except:
                        print(x,j)
                a=list(set(colj))  # posible values in jth dimension
                self.ConditProb[i].append([])   # store the jth dimension of prob (dim,len(a))
                for l in range(len(a)):
                    '''
                    now in data find caculate the condition prob
                    '''
                    Denominator=1e-10;Numerator=0.0
                    for m in range(self.num):
                        if self.label[m]==classes[i]: # Y=Ck
                            Denominator+=1
                            if self.data[m][j]==a[l]: # Xj=al ^Yi=Ck
                                Numerator+=1
                    self.ConditProb[i][j].append(Numerator/Denominator)
                    print('class:',classes[i],'dim:',j,'al:',a[l],'prob:',Numerator/Denominator)
        
    def predict(self,X):
        K=len(self.PriorProb)
        dim=len(X)
        prob=[]
        for i in range(K):
            condictProb=1.0
            for j in range(dim):
                colj=[]
                for x in range(self.num):
                    colj.append(self.data[x][j])
                a=list(set(colj))  # posible values in jth dimension
                for l in range(len(a)):
                    if X[j]==a[l]:
                        condictProb*=self.ConditProb[i][j][l]                      
            prob.append(self.PriorProb[i]*condictProb)
       
        max_score=max(prob)
        max_index=prob.index(max_score)
        classes=list(set(self.label))
        return max_score,classes[max_index]
    
    
    
###### test
if __name__=='__main__':
    data=[[1,'S'],
          [1,'M'],
          [1,'M'],
          [1,'S'],
          [1,'S'],
          [2,'S'],
          [2,'M'],
          [2,'M'],
          [2,'L'],
          [2,'L'],
          [3,'L'],
          [3,'M'],
          [3,'M'],
          [3,'L'],
          [3,'L']]
    
    label=[-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]
    X=[2,'S']
  
    data1=[[6, 180, 12],
           [5.92, 190, 11],
           [5.58, 170, 12],
           [5.92, 165, 10],
           [5, 100, 6],
           [5.5, 150, 8],
           [5.42, 130, 7],
           [5.75, 150, 9]]
    
    label1=[1, 1, 1, 1, 0, 0, 0, 0]
    X1=[6, 130, 8]
  
    
    
    bayes=NaiveBayes(data1,label1)
    bayes.caculProb()
    score,clas=bayes.predict(X1)
    print('class:',clas)
    print('score:',score)
    print('the data1 show us that same of data may cause peob==0,we should use Bayesian estimation')
    
                        
            
        
                    
                                
                            
                        
                    
                    
        
        
        
        
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        