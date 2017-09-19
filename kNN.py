#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:26:33 2017

@author: dhingratul
"""
import pickle
import numpy as np


class NearestNeighbors(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_te):
        num = X_te.shape[0]
        y_pred = np.zeros(num, dtype=self.y_train.dtype)
        for i in range(num):
            distances = np.sum(np.abs(self.X_train - X_te[i, :]), axis=1)
            min_index = np.argmin(distances)
            y_pred = self.y_train[min_index]
        return y_pred


def unpickle(file):
    # unpickles cifar10 dataset
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def getCIFAR10(direc, filename, batches):
    for j in range(1, batches+1):
        file = direc + filename + str(j)
        dic = unpickle(file)
        if j == 1:
            X_train = dic[b'data']
            y_train = dic[b'labels']
        else:
            temp_X = dic[b'data']
            temp_y = dic[b'labels']
            X_train = np.concatenate((X_train, temp_X))
            y_train = np.concatenate((y_train, temp_y))
    return X_train, y_train


# Driver
direc = '/home/dhingratul/Documents/Dataset/cifar-10-batches-py/'
test_file = 'test_batch'
filename = 'data_batch_'
X_train, y_train = getCIFAR10(direc, filename, 5)
data_test = unpickle(direc + test_file)
X_test = data_test[b'data']
y_test = data_test[b'labels']
# Call NN
nn = NearestNeighbors()
nn.train(X_train, y_train)
Yte_predict = nn.predict(X_test)
print('accuracy: {}' .format(np.mean(Yte_predict == y_test)))
