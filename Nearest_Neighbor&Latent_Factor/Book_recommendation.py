#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:36:35 2017

@author: raghav
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import time
import tensorflow as tf
DEVICE="/gpu:0"

data = pd.read_csv('BX-Book-Ratings.csv',sep=";",encoding="ISO-8859-1",
                    header=0,names=['user','isbn','rating'])

print(data.head())

book = pd.read_csv('BX-Books.csv',sep=";",encoding="ISO-8859-1",
                    header=0,error_bad_lines=False,usecols=[0,1,2],index_col=0,
                    names=['isbn','title','author'])
print (book.head())



#               Nearest neighbor implementation         #

def bookMeta(isbn):
    title = book.at[isbn,'title']
    author = book.at[isbn,'author']
    return title,author

#return table for top N recommendations for user
def favebooks(user,N):
    userRatings = data[data["user"]==user]
    sortedRatings = pd.DataFrame.sort_values(userRatings,['rating']
                ,ascending=[0])[:N]
    sortedRatings['title']=sortedRatings['isbn'].apply(bookMeta)
    return sortedRatings

data = data[data['isbn'].isin(book.index)]# To avoid any inconsistency between
#both data and book dataframes such that isbn number mapping is consistent
"""
favebooks(276762,5)
"""
userPerISBN = data.isbn.value_counts()# unique values of isbn
print (userPerISBN.shape)

ISBNperUser = data.user.value_counts()
print (ISBNperUser.shape)

# To reduce sparsity we take only those isbn into account which are 
# read by more than 10 users

data = data[data['isbn'].isin(userPerISBN[userPerISBN>10].index)]
print (data.shape)
u  = len(data['user'])
i = 3
# And the users qhich have read more than 10 books

data = data[data['user'].isin(ISBNperUser[ISBNperUser>10].index)]

# Creating the rating matrix 

userItemRatingMatrix = pd.pivot_table(data,values='rating',index=['user'],
                                      columns=['isbn'])
# values parameter = values which will be used to fill the matrix
# index parameter = index for rows
# columns parameter = index for columns

print(userItemRatingMatrix.head())


user1 = 204622
user2 = 255489

# Pandas allows us to pick a column from a 2d matrix but not a row.
# So we pick up the row by transposing the matrix and picking up the column

user1Ratings = userItemRatingMatrix.transpose()[user1]
user2Ratings = userItemRatingMatrix.transpose()[user2]
"""
"""
from scipy.spatial.distance import hamming
#print(hamming(user1Ratings,user2Ratings))

def distance(user1,user2):
    try:
        user1Ratings = userItemRatingMatrix.transpose()[user1]
        user2Ratings = userItemRatingMatrix.transpose()[user2]
        distance = hamming(user1Ratings,user2Ratings)
    except:
        distance = np.NaN
    return distance

print (distance(204622,255489))

def nearestNeighbors(user,k=10):
    allUsers = pd.DataFrame(userItemRatingMatrix.index)
    allUsers = allUsers[allUsers.user!=user]
    allUsers['distance'] = allUsers['user'].apply(lambda x: distance(user,x))
    KnearestUsers = allUsers.sort_values(['distance'],ascending=True)['user'][:k]
    return KnearestUsers

user=204622# This is a sample active user

def topN(user,N=3):
    KnearestUsers = nearestNeighbors(user)
    #print (BooksAlreadyRead)
    NNratings = userItemRatingMatrix[userItemRatingMatrix.index.isin(KnearestUsers)]
    #NNratings are the ratings given by user's nearest neighbors neighbors 
    avgRating = NNratings.apply(np.nanmean).dropna()
    #nanmean returns mean column by column dropping nan values
    BooksAlreadyRead = userItemRatingMatrix.transpose()[user].dropna().index
    avgRating = avgRating[~avgRating.index.isin(BooksAlreadyRead)]
    # Suggesting top 3 recommendations
    topRecommend = avgRating.sort_values(ascending =False,).index[:N]
    return pd.Series(topRecommend).apply(bookMeta)

# Testing the code
favebooks(204813,10)
topN(204813,10)


                ###             Latent factor analysis      ###

from scipy.sparse import coo_matrix

#coo_matrix((values,(rowsource,columnsource)))
#The parameters values,rowsource and columnsource should be integer values
# so we internally map 'user' and 'isbn' as integers for row and columnsource
data['user'] = data['user'].astype('category')
data['isbn'] = data['isbn'].astype('category')

R = coo_matrix((data['rating'].astype(float),(data['user'].cat.codes.copy(),
               data['isbn'].cat.codes.copy())))

#Here R is a matrix having a attribute data that has only rated values as array
#R has same dimension as userRatingMatrix 
print (R.shape)

#R.data contains only non-null values
print (len(R.data))

#                   Converting to tf coo_matrix
"""
tf_coo_matrix = tf.SparseTensor(indices = np.array([R.row,R.col]).T,
    values=R.data,dense_shape=R.shape)
node = tf_coo_matrix.values[4]
#Note: tf_coo_matrix.values returns a 1d tensor
node1 = (tf.shape(tf_coo_matrix))
node2 = (tf.shape(tf_coo_matrix))[1]
row = tf_coo_matrix._indices[4][0]
col = tf_coo_matrix._indices[4][1]
node3 = tf_coo_matrix.dense_shape
sess = tf.Session()
print (sess.run(node))
print (sess.run(node1))
print (sess.run(node2))
print (sess.run(row))
print (sess.run(col))
print (sess.run(node3))


print (R.data[4])
print (R.row[4])
print (R.col[4])
"""

#Accessing the data array
"""
R.data[0] -The rating value
R.row[0] -The corresponing row number
R.col[0] -The corresponding col number
"""

#             Python implementation of latent factor      #

"""
# Initialising factor matrix
M,N = R.shape
print (R.shape)
# Here M= no. of users and N=no. of products
K=3
#We keep 3 factors i.e underlying factors driving the ratings

#Now we randomly initialise the P and Q matrix and try minimising the error
P= np.random.rand(M,K)
Q= np.random.rand(K,N)

from numpy.linalg import norm

# We write the error function on our own to achieve the optimistaion objective
def error (R,P,Q,lamda=0.02):
    ratings = R.data
    rows = R.row
    cols = R.col
    e=0;
    for ui in range(len(ratings)):
        rui = ratings[ui]
        u = rows[ui]
        i = cols[ui]
        if (rui>0):
            e = e + pow( (rui - np.dot(P[u,:],Q[:,i])) , 2) + lamda * ( pow(norm(P[u:,]),2) + pow(norm(Q[:,i]),2) ) 
    return e

# error function is total squared error wich we want to minimise
# rmse is root mean squared error
print(error(R,P,Q))

rmse = np.sqrt(error(R,P,Q)/len(R.data))

# Implementing SGD to minimise rmse in python(Training time is very high, Use TF)

def SGD(R,K,lamda =0.02, steps=10,gamma = 0.001):
    # K is the number of factors
    # lamda is the regularisation parameter
    # Steps is the max number of iterations allowed before finalising p and q
    # gamma is the size of the step or learning rate alpha
    M,N = R.shape
    P = np.random.rand(M,K)
    Q = np.random.rand(K,N)
    
    rmse = np.sqrt(error(R,P,Q,lamda)/len(R.data))
    print ('Initial rmse =' + str(rmse))
    for step in range(steps):
        for ui in range(100):
            rui = R.data[ui]
            u = R.row[ui]
            i= R.col[ui]
            if (rui>0):
                startTime = time.time()
                eui = rui - np.dot(P[u,:],Q[:,i])
                P[u,:] = P[u,:] + gamma*2* (eui*Q[:,i] - lamda*P[u,:])
                Q[:,i] = Q[:,i] + gamma*2* (eui*P[u,:] - lamda*Q[:,i])
                endTime = time.time();
                print ('Iteration %d completed in %0.5fs' %(ui,endTime-startTime))
            rmse = np.sqrt(error(R,P,Q)/len(R.data))
        if (rmse<0.5):
            break
    print ('final rmse = ' + str(rmse))
    return P,Q

STime = time.time()
P,Q = SGD(R,K=2,gamma=0.0007,lamda=0.01,steps=1)
ETime = time.time()
print ('Total time taken=%0.5fs' %(ETime-STime))
"""             

"""
K=2
M,N = R.shape
W = tf.Variable(tf.truncated_normal([M,K],mean=0,stddev=0.2),name='users')
H = tf.Variable(tf.truncated_normal([K,N],mean=0,stddev=0.2),name='items')
result = tf.matmul(W,H)
mera_session = tf.Session()
mera_session.run(tf.global_variables_initializer())
print (mera_session.run(result))

a=list()
tf_coo_matrix = tf.SparseTensor(indices = np.array([R.row,R.col]).T,
    values=R.data,dense_shape=R.shape)

rw = list()
cl = list()
for i in range(0,len(R.data)):
    rw.append(R.row[i])
    cl.append(R.col[i])
"""
# This factorizer is without user bias and item bias and mean addition

def factorizer(R,K):
    M,N = R.shape
    W = tf.Variable(tf.truncated_normal([M,K],mean=2,stddev=1),name='users')
    H = tf.Variable(tf.truncated_normal([K,N],mean=2,stddev=1),name='items')
    result = tf.matmul(W,H)
    ses = tf.Session()
    ses.run(tf.global_variables_initializer())
    initial_matrix = result
    print ('Result:')
    print (ses.run([initial_matrix]))
    
    tf_coo_matrix = tf.SparseTensor(indices = np.array([R.row,R.col]).T,
    values=R.data,dense_shape=R.shape)
    
    rw = list()
    cl = list()
    for i in range(0,len(R.data)):
        rw.append(R.row[i])
        cl.append(R.col[i])
    
    #Now we want to know only those values which were known earlier in user-item matrix
    result_values = tf.gather(tf.reshape(result, [-1]),rw * tf.shape(result)[1]+ cl,
                              name="extract_training_ratings")
    diff_op = tf.subtract(result_values,tf.cast(tf_coo_matrix.values,tf.float32),name = 'raw_train_error')
    
    lda = 0.02
    lr = 1.3
    num_ratings = len(R.data)
    with tf.name_scope('training_cost') as scope:
        base_cost = tf.reduce_sum(tf.square(diff_op,name='squared_difference'),name='sum_squared_error')
        regularizer = tf.multiply(tf.add(tf.reduce_sum(tf.square(W)),tf.reduce_sum(tf.square(H))),lda,name='regularize')
        cost = tf.div(tf.add(base_cost,regularizer),2*num_ratings,name='average_error')
    
    #We use an exponentially decaying learning rate
    global_step = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(lr,global_step,10000, 0.96, staircase=True)
   
    with tf.name_scope('train') as scope:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Passing global_step to minimize() will increment it at each step so
        # that the learning rate will be decayed at the specified intervals.
        train_step = optimizer.minimize(cost, global_step=global_step)
    
    """threshold = 0.5
    with tf.name_scope("training_accuracy") as scope:
        # Just measure the absolute difference against the threshold
        good = tf.less(tf.abs(diff_op), threshold)
        accuracy_tr = tf.div(tf.reduce_sum(tf.cast(good, tf.float32)), num_ratings)
    """
    
    sess = tf.Session() 
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    startTime = time.time();
    for i in range(10):
        if( i%10 == 0):
            endTime = time.time()
            print ('Iteration %d completed in %0.5fs' %(i,endTime-startTime))
            startTime = endTime
            res = sess.run([cost])
            cost_ev = res[0]
            #print("Training accuracy at step %s: %s" % (i, acc_tr))
            print("Training cost: %s" % (cost_ev))
        else:
            sess.run(train_step)
    
    final_matrix = result#Note: add mean rating
    final_res = sess.run([final_matrix])
    print (final_res)
    saver.save(sess,'my_model')
    sess.close()
    return (final_res,saver)       
    
    
stime = time.time()   
ans_matrix,saver = factorizer(R,2)
print ('ans_matrix:')
print (ans_matrix)
etime = time.time()
print ("Total time taken = %0.5fs" %(etime-stime))
sess = tf.Session()

"""
new_saver = tf.train.import_meta_graph('my_model.meta')
new_saver.restore(sess,tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name('final_matrix')
print (sess.run(final_res))
all_vars = tf.get_collection('vars')
for v in all_vars:
    v_ = sess.run(v)
    print(v_)
#floyd logs -t Z5qyVEPMKESQESuj3Ug489
#data ID: ehd4KUgzrWim3mjgFcnK37
"""