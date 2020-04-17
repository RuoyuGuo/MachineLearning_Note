# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:53:02 2019

@author: Ruoyu
"""

import numpy as np
import pandas as pd

def std_norm(train_x, test_x):
    mean = np.mean(train_x, 0)
    std = np.std(train_x, 0)
    
    train_x = (train_x - mean)/std
    test_x = (test_x - mean)/std
    
    return train_x, test_x

def training(train_x, train_y, re_weight):
    w = np.ones((106, ), dtype=float)
    b = 1
    lr = 0.2
    iteration = 100
    ada_w = np.zeros((106, ), dtype=float)
    ada_b = 0
    
    for T in range(iteration):
        w_grad = np.zeros((106, ), dtype=float)
        b_grad = 0
        
        print(T)
        for i in range(train_x.shape[0]):
            w_grad += -(train_y[i] - 1/(1 + np.exp(-w.dot(train_x[i]) - b)))\
                       * (train_x[i]) + 2 * w * re_weight
            b_grad += -(train_y[i] - 1/(1 + np.exp(-w.dot(train_x[i]) - b)))
            
        
        ada_w += (w_grad ** 2)
        ada_b += (b_grad ** 2)
        
        w = w - lr/np.sqrt(ada_w) * w_grad
        b = b - lr/np.sqrt(ada_b) * b_grad
    
    print(f'w = {w}')
    print(f'b = {b}')
    
    np.save('weight', w)
    np.save('bias', b)
    
def model(x, w, b):
    pro = 1/(1 + np.exp(-w.dot(x) - b))
    
    pro = np.clip(pro, 1e-6, (1 - 1e-6))
    
    if pro >= 0.5:
        return 1
    else:
        return 0
        
    
if __name__ == '__main__':
    
    #--------------------load  data------------------
    train_x = np.loadtxt('X_train', delimiter=',', skiprows=1)
    train_y = np.loadtxt('Y_train', delimiter=',', skiprows=1)  
    test_data = np.loadtxt('X_test', delimiter=',', skiprows=1)
    
    train_x, test_data = std_norm(train_x, test_data)
    
    #--------------------training--------------------
    training(train_x, train_y, 0.1)
    
    #-------------------predict---------------------------
    output_id = np.arange(1, test_data.shape[0]+1, dtype=int)
    output_label = np.empty((test_data.shape[0],), dtype=int)
    w = np.load('weight.npy')
    b = np.load('bias.npy')
    
    for i in range(test_data.shape[0]):
        output_label[i] = model(test_data[i], w, b)
    
    df_predicted = pd.DataFrame({'id': output_id,
                             'label': output_label
                             })
    df_predicted.to_csv('predicted_output.csv', index=False)