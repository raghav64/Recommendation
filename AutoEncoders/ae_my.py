#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:18:24 2017

@author: raghav
"""

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

#Creating architecture of Autoencoders

class SAE(nn.Module):
    def __init__(self,):
        super(SAE,self).__init__()#So that we have all variables inherited from nn class
        self.fc1 = nn.Linear(nb_movies,20)#Using torch.nn for introducing layers
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20,nb_movies)
        self.activation = nn.Sigmoid()
    
    def forward(self,x):
        x = self.activation(self.fc1(x))#This returns encoded vector after
        #encoding of first full connection layer
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)#No activation needed when output is to be generated
        return x

sae = SAE()
criterion = nn.MSELoss()
optimiser = optim.RMSprop(sae.parameters(), lr= 0.01 ,weight_decay=0.5)#weight_decay
# will decay the lr exponentially

#Training the SAE

nb_epoch = 200
for epoch in range(1,nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)#Torch does not 
        #take single input vector as input but rather requires a batch_size
        #as 1st argument
        target = input.clone()
        if (torch.sum(target.data > 0 )>0):#Eliminate the user with no ratings
            output = sae(input)
            target.require_grad = False#Faster computation as target remains unchanged
            output[target==0] = 0#So that unrated movie dont account for any error
            loss = criterion(output,target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0 ) + 1e-10)
            #mean_corrector is to adjust the non zero ratings
            loss.backward()#Decides the direction in which the weights will be updated
            train_loss += np.sqrt(loss.data[0]*mean_corrector)# To adjust the relevant loss
            s+=1.
            optimiser.step()#Decides the intensity of weight changes
    print('epoch: '+ str(epoch) + 'loss: '+ str(train_loss/s))
            
        
    test_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) 
        #We need to take input as training_set input as the reconstruction and
        #weight activation will take place on basis of training input and then
        #the reconstructed output will be compared to test_file to compute loss
        target = Variable(test_set[id_user])
        #So target here is test_file
        if (torch.sum(target.data > 0 )>0):#Eliminate the user with no ratings
            output = sae(input)
            target.require_grad = False#Faster computation as target remains unchanged
            output[target==0] = 0#So that unrated movie dont account for any error
            loss = criterion(output,target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0 ) + 1e-10)
            #mean_corrector is to adjust the non zero ratings
            test_loss += np.sqrt(loss.data[0]*mean_corrector)# To adjust the relevant loss
            s+=1.
    print('Test loss: '+ str(test_loss/s))
              
        





