#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:42:57 2017

@author: dhingratul
Run file for k- Nearest neighbor algorithm
"""
import numpy as np
import kNN as kNN

# Driver
direc = '../data/'
test_file = 'test_batch'
filename = 'data_batch_'
X_train, y_train = kNN.getCIFAR10(direc, filename, 5)
data_test = kNN.unpickle(direc + test_file)
X_test = data_test[b'data']
y_test = data_test[b'labels']
# Call NN
knn = kNN.kNearestNeighbors()
k = 5
knn.train(X_train, y_train, k)
Yte_predict = knn.predict(X_test)
print('accuracy: {}' .format(np.mean(Yte_predict == y_test)))
