#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:01:25 2017

@author: raghav
"""
#Importing the Libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#Importing the dataset

movies = pd.read_csv('ml-1m/movies.dat',sep='::',header= None 
                     ,engine='python',encoding='latin-1')
#The separator is :: not the , and we have no column names so header=None
#and encoding = latin-1 so that special characters in the movie names dont get ignored
users = pd.read_csv('ml-1m/users.dat',sep='::',header= None 
                     ,engine='python',encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep='::',header= None 
                     ,engine='python',encoding='latin-1')

#Preparing the training and test set

#We will take training and test from ml-100k dataset i.e base and test files
training_set = pd.read_csv('ml-100k/u1.base',delimiter='\t')
#u1.base is a tab separated file so instead of sep we use delimiter parameter
#The training_set obtained above is a dataframe. For using pytorch it 
#is necessary to convert into array datatype using np 
training_set = np.array(training_set , dtype='int')
#Similar for test_set
test_set = pd.read_csv('ml-100k/u1.test',delimiter='\t')
test_set = np.array(test_set , dtype='int')

#Getting the number of users and movies for UI matrix

#The max user ID can lie in either train or test dataset
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

#Converting the data into 2d-array UI matrix
#We convert the data to list of lists rather than a 2d np array cuz pytorch needs

def convert(data):
    new_data = []
    for id_users in range(1,nb_users+1):
        id_movies = data[:,1][data[:,0]==id_users]
        id_ratings = data[:,2][data[:,0]==id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        #id_movies are 1-indexed
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

#Converting the data to torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1(liked) or 0(not liked)
# because we want our prediction to be binary too.
training_set[training_set == 0] = -1
#Setting the unrated movies as -1 and with rating 1 or 2 as 0 and others as 1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the neural network RBM class

class RBM():
    
    def __init__(self,nv,nh):#All the variables defined in init are object specific.If defined above then the variable will be global and accessible to all objects.      
        self.W = torch.randn(nh,nv)
        self.a = torch.randn(1,nh)#bias for hidden layer. Because torch accepts 2d 
        #tensors only where first dimension corresponds to batch_size so randn is 1,nh
        self.b = torch.randn(1,nv)#bias for visible layer.1 corresponds to batch_size
    
    #We define a sample function which will sample hidden nodes based on their 
    #activation i.e which neurons fire given conditional probability(sigmoid activation):
    #prob of h firing given v
    #x represents the visible layer neurons with dimensions 1*nv
    #y represents the hidden layer neurons with dimensions 1*nh
    #W represents 2d torch tensor nh*nv
    def sample_h(self,x):
        wx = torch.mm(x,self.W.t())#dimension 1*nh
        activation = wx + self.a.expand_as(wx)#As the bias a is to be fit in 
        #each line of mini_batch so it needs to be expanded.
        p_h_given_v = torch.sigmoid(activation)
        # We return the bernoulli samples by selecting a random number b/w
        #0 and 1 and comparing it with p h/v.This is handled by torch function bernoulli
        return p_h_given_v,torch.bernoulli(p_h_given_v)
    
    def sample_v(self,y):
        wy = torch.mm(y,self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h,torch.bernoulli(p_v_given_h)
    
    #We train the RBM using contrastive divergence using Gibbs sampling.
    #Find the algorithm used here in the research paper in this folder. 
    
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)
        self.b += torch.sum((v0-vk),0)
        self.a += torch.sum((ph0-phk),0)
       
# Creating an RBM class object
nv = len(training_set[0])#nv(num of visible nodes) is number of movies.
nh = 100
batch_size = 100#We chose the batch_size as 1 in init. We can change it to
#take argument batch_size so that training can be done in batches.
#Also this is needed for training below
rbm = RBM(nv,nh)

#Training the RBM

nb_epoch = 10
for epoch in range(1,nb_epoch+1):
    train_loss = 0
    s=0.#Just a counter to normalise the train_loss by dividing the loss by num_of_batches
    for id_users in range(0,nb_users-batch_size,batch_size):
        vk = training_set[id_users:id_users+batch_size]
        v0 = training_set[id_users:id_users+batch_size]#These are target ratings
        #which wont get changed
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):#For k step contrastive divergence in Gibbs-Sampling
            _,hk = rbm.sample_h(vk)#hidden nodes after k steps
            _,vk = rbm.sample_v(hk)#visible nodes after k steps
            vk[v0<0] = v0[v0<0]#Not to update the -1 ratings in training
        phk,_ = rbm.sample_h(vk)
        
        rbm.train(v0,vk,ph0,phk)
        
        train_loss+=torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
        s+=1.
    print ('epoch: ' + str(epoch)+ ' loss: '+str(train_loss/s))
        
# Testing the RBM

    test_loss = 0
    s=0.#Just a counter to normalise the train_loss by dividing the loss by num_of_batches
    for id_users in range(nb_users):
        v = training_set[id_users:id_users+1]
        vt = test_set[id_users:id_users+1]#These are target ratings
        #which wont get changed
        if(len(vt[vt>=0])):
            _,h = rbm.sample_h(v)
            _,v = rbm.sample_v(h)
            test_loss+=torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
            s+=1.
    print ('epoch: '+ ' loss: '+str(test_loss/s))
    
    
        
        
        


