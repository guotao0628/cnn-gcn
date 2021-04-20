'''
Created on 2018-11-10

@author: 南城
'''
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random
# import get_data 

import Normal
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data():
    adj,features,labels=Normal.loadData()

    idx_train = [i for i in range(615)]
    idx_val=[i for i in range(615,875)]
    idx_test=[i for i in range(615,875)]
#     idx_train=[i for i in num if i not in idx_test]
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
#    
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
#     
    return adj, features, y_train,y_val,y_test, train_mask,val_mask,test_mask
# load_data()
