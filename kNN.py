#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:26:33 2017

@author: dhingratul
Helper file for Nearest Neighbor algorithm
"""
import pickle
import numpy as np
from collections import Counter


class NearestNeighbors(object):
    # Nearest neighbor classifier
    def __init__(self):
        pass

    def train(self, X, y):
        # Learn the training instances
        self.X_train = X
        self.y_train = y

    def predict(self, X_te):
        num = X_te.shape[0]
        y_pred = np.empty_like(self.y_train)
        for i in range(num):
            distances = np.sum(np.abs(self.X_train - X_te[i, :]), axis=1)
            min_index = np.argmin(distances)
            y_pred[i] = self.y_train[min_index]
        return y_pred


class kNearestNeighbors(object):
    # Nearest neighbor classifier
    def __init__(self):
        pass

    def train(self, X, y, k):
        # Learn the training instances
        self.X_train = X
        self.y_train = y
        self.k = k

    def predict(self, X_te):
        num = X_te.shape[0]
        y_pred = np.empty_like(self.y_train)

        for i in range(num):
            L = []
            distances = np.sum(np.abs(self.X_train - X_te[i, :]), axis=1)
            min_indices = np.argsort(distances)
            for j in range(self.k):
                L.append(self.y_train[min_indices[j]])
            y_pred[i] = Counter(L).most_common(1)[0][0]
        return y_pred


def unpickle(file):
    # unpickles cifar10 dataset
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def getCIFAR10(direc, filename, batches):
    # Converts the data in batches to a full training set
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
