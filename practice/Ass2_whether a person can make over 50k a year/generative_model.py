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
    
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    return train_x, test_x

def min_max_norm(train_x, test_x):
    train_min = np.min(train_x)
    train_max = np.max(train_x)
    
    train_x = (train_x - train_min) / (train_max - train_min)
    test_x = (test_x - train_min) / (train_max - train_min)
    
    return train_x, test_x

def mean_norm(train_x, test_x):
    train_min = np.min(train_x)
    train_max = np.max(train_x)
    mean = np.mean(train_x, 0) 
    
    train_x = (train_x - mean) / (train_max - train_min)
    test_x = (test_x - mean) / (train_max - train_min)
    
    return train_x, test_x

def unit_length_norm(train_x, test_x):
    eulidean_length = np.linalg.norm(train_x)
    
    train_x = train_x / eulidean_length
    test_x = test_x / eulidean_length
    
    return train_x, test_x

def training(train_x, train_y):
#c1 personal incoming over 50k per year, label 1
#c2 personal incoming under 50k per year, label 0
    train_x_c1 = train_x[train_y == 1]
    train_x_c2 = train_x[train_y == 0]
    
    total_num_c1 = train_y[train_y == 1].shape[0]
    total_num_c2 = train_y[train_y == 0].shape[0]
    
    mean_c1 = np.mean(train_x_c1, 0).reshape(1, -1)
    mean_c2 = np.mean(train_x_c2, 0).reshape(1, -1)
    
    '''
    covar_c1 = (train_x_c1 - mean_c1).T.dot(train_x_c1 - mean_c1)/total_num_c1
    covar_c2 = (train_x_c2 - mean_c2).T.dot(train_x_c2 - mean_c2)/total_num_c2
    '''
    #
    
    covar_c1 = np.zeros((106, 106), dtype=float)
    covar_c2 = np.zeros((106, 106), dtype=float)
    
    for v in train_x_c1:
        v = v.reshape(1, -1)
        covar_c1 += (v - mean_c1).T.dot(v-mean_c1)/total_num_c1
        
    for v in train_x_c2:
        v = v.reshape(1, -1)
        covar_c2 += (v - mean_c2).T.dot(v - mean_c2)/total_num_c2
        
    co_var = ((total_num_c1 * covar_c1) + (total_num_c2 * covar_c2))\
                 /(total_num_c1 + total_num_c2)
               
    def model(x):
    #Gaussian distribution
        
        w = (mean_c1 - mean_c2).dot(np.linalg.inv(co_var))
    
        b = (-0.5) * mean_c1.dot(np.linalg.inv(co_var)).dot(mean_c1.T) \
            + 0.5 * mean_c2.dot(np.linalg.inv(co_var)).dot(mean_c2.T) \
            + np.log(total_num_c1/total_num_c2)
            
        pro = 1/(1 + np.exp( - w.dot(x) - b))
        
        if pro > 0.5:
            return 1
        
        else:
            return 0
        '''
        x = x.reshape(1, -1)
        
        part1_p_x_c1 = np.exp((-0.5 * (x - mean_c1).dot(np.linalg.inv(co_var)).dot((x - mean_c1).T)))
        
        part1_p_x_c2 = np.exp((-0.5 * (x - mean_c2).dot(np.linalg.inv(co_var)).dot((x - mean_c2).T)))
        
        part2 = np.sqrt(np.power((2 * np.pi), 106) * np.linalg.det(co_var))
        
        p_x_c1 = part1_p_x_c1 / part2
        p_x_c2 = part1_p_x_c2 / part2
        
        p_c1 = total_num_c1 / (total_num_c1 + total_num_c2)
        p_c2 = total_num_c2 / (total_num_c1 + total_num_c2)
        
        pro = p_x_c1 * p_c1 / (p_x_c1 * p_c1 + p_x_c2 * p_c2)
        '''
        if pro > 0.5:
            return 1 
        else:
            return 0
          
    return model
       
if __name__ == '__main__':
    
    #--------------------load training data------------------
    train_x = np.loadtxt('X_train', delimiter=',', skiprows=1)
    train_y = np.loadtxt('Y_train', delimiter=',', skiprows=1)
    
    #-------------------load testing data--------------------
    test_data = np.loadtxt('X_test', delimiter=',', skiprows=1)
    
    #------------------normalization----------------
    train_x, test_data = unit_length_norm(train_x, test_data)
    
    #--------------------training--------------------
    model = training(train_x, train_y)
        
    #-------------------predict---------------------------
    output_id = np.arange(1, test_data.shape[0]+1, dtype=int)
    output_label = np.zeros((test_data.shape[0],), dtype=int)
    
    for i in range(test_data.shape[0]):
        output_label[i] = model(test_data[i])
    
    df_predicted = pd.DataFrame({'id': output_id,
                             'label': output_label
                             })
    
    df_predicted.to_csv('predicted_output.csv', index=False)
    