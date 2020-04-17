# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:16:25 2019

@author: Ruoyu
"""

import pandas as pd
import numpy as np

def model_fun1(train_x, train_y, w, b, re_w):
#y = W1X1 + W2X2 +...WnXn + b
#seen 1 feature * 9 (hours) as functions' features
    w_grad = 2 * (train_y - b - train_x.dot(w) ) * (-train_x) + 2 * re_w * w
    b_grad = 2 * (train_y - b - train_x.dot(w) ) * (-1)
    
    return w_grad, b_grad


def preprocess_train_data(train_set):
    months = 12
    days = 20
    hours = 24  
    num_of_fea = 18
    
    #convert train_set shape to (18, ...)
    np_data = np.empty((num_of_fea, months * days * hours), dtype = float)
    
    for m in range(months):
        for d in range(days):           
            for f in range(num_of_fea):
                for h in range(hours):
                    np_data[f, m * days * hours + d * hours + h] = \
                       train_set[m * days * num_of_fea + d * num_of_fea + f, h]
    
    '''
    #cross validation
    num_of_col = np_data.shape[1]
    np_train_data = np_data[:, : int(num_of_col * 2 / 3)]    #ntd    2/3 of the train_set
    np_valid_data = np_data[:, int(num_of_col * 2 / 3):]     #nvd    rest 1/3 of the train_set
    '''
    np_train_data = np_data
    
    #extract input and real output
    #9-hrs data   shape : 1, 1 * 9
    #10th-hr pm2.5   shape: 1, 1
    train_x = np.empty((12 * (480 - 9), 1 * 9), dtype = float)                
    train_y = np.empty((12 * (480 - 9), ), dtype = float)                

    
    #for each continuous 10 hours, use the every 9 hours' pm2.5 in a month as x
    #and the 10th hour's pm2.5 value as the real y
    for m in range(months):
        for d in range(days):
            for h in range(hours):
                    if d == 19 and h > 14:
                        continue
                    train_x[m * (480 - 9) + d * 24 + h] = \
                        np_train_data[9, m * 480 + d * 24 + h:
                                     m * 480 + d * 24 + h + 9].reshape(1, -1)
                    train_y[m * (480 - 9) + d * 24 + h] = \
                        np_train_data[9, m * 480 + d * 24 + h + 9]
    '''
    valid_x = []
    valid_y = []
    
    
    #work with valid data
    for i in range(len_of_nvd - 9):
        valid_x.append(np_valid_data[:, i:i + 9]).reshape(1, -1)
        valid_y.append(np_valid_data[9, i + 9])
    '''
    
    return train_x, train_y#, valid_x, valid_y

def my_rmse(y_hat, train_x, w, b):
    pre_y = np.empty((y_hat.shape[0], ))
    
    for i in range(pre_y.shape[0]):
        pre_y[i] = train_x[i].dot(w) + b
    
    return np.sqrt(np.mean( (y_hat - pre_y) ** 2))


#model1 1 * 9 features
def training(model_fun, train_x, train_y):
    #------------linear regression------------
    w = np.ones((1 * 9, ))
    b = 1
    lr = 200
    iteration = 5000
    #bias = 0
    #variance = 0
    
    #record w and b per updating
    w_history = []
    b_history = []
    
    ada_w = np.ones((1 * 9, ))
    ada_b = 1
    
    for T in range(iteration):
        w_grad = np.zeros((1 * 9, ))
        b_grad = 0
        
        print(T)
        
        
        #compute partial derivativte of parameters
        for i in range(len(train_x)):
            #no regularization
            #t_w_grad, t_b_grad = model_fun(train_x[i], train_y[i], w, b, 0)  
            #lambda = 0.1
            t_w_grad, t_b_grad = model_fun(train_x[i], train_y[i], w, b, 10)
            #lambda = 0.01
            #t_w_grad, t_b_grad = model_fun(train_x[i], train_y[i], w, b, 0.01)
            #lambda = 0.001
            #t_w_grad, t_b_grad = model_fun(train_x[i], train_y[i], w, b, 0.001)
            #lambda = 0.0001
            #t_w_grad, t_b_grad = model_fun(train_x[i], train_y[i], w, b, 0.0001)      
            w_grad = w_grad + t_w_grad
            b_grad = b_grad + t_b_grad
            
        #w_history.append(w_grad)
        #b_history.append(b_grad)
        
        #sum of square
        ada_w = ada_w + w_grad ** 2
        ada_b = ada_b + b_grad ** 2
        
        w = w - lr/np.sqrt(ada_w) * w_grad   
        b = b - lr/np.sqrt(ada_b) * b_grad
        
        w_history.append(w)
        b_history.append(b)
   
    
    #------------output coefficient of function------------
    print(f'w = {w}')
    print()
    print(f'b = {b}' )
    print()
    
    
    #------------output statistics------------------
    print(f'RMSE = {my_rmse(train_y, train_x, w, b)}')
    #print(f'bias = {bias}')
    #print(f'variance = {variance}')
    
    
    np.save('weight', w)
    np.save('bias', b)
    

#model 1 output
def predict():
    w = np.load('weight.npy')
    b = np.load('bias.npy')
    
     #------------extract test data------------------------
    raw_test_set = pd.read_csv('test.csv', header=None)
    test_x = raw_test_set.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')\
                           .fillna(0.0)\
                           .to_numpy()
    
    num_of_test = int(test_x.shape[0]/18)
    
    output_id = [f'id_{i}' for i in range(num_of_test)]
    output_pm2_5 = np.empty(num_of_test)


    #----------compute predicted pm2.5 of each data----------------
    for i in range(num_of_test):
        input_features = test_x[9 * (i + 1), : ].reshape(1, -1)
        pm2_5 = np.dot(input_features[0], w) + b
        output_pm2_5[i] = pm2_5


    #save to a dataframe
    dict_output = {'id': output_id, 'pm2.5': output_pm2_5}
    df_output = pd.DataFrame(dict_output)
    
    
    #save as csv file
    df_output.to_csv('predicted_output.csv', header = ['id', 'value'], 
                        index = False)


if __name__ == "__main__":

    
    #-----------------------extract train data------------------------------
    raw_train_set = pd.read_csv('train.csv')
    train_set = raw_train_set.iloc[:, 2:].apply(pd.to_numeric, errors = 'coerce')\
                                         .fillna(0.0)\
                                         .to_numpy()
    
    
    #train_x: every row is a sample
    #preprocess trainning data
    train_x, train_y = preprocess_train_data(train_set)
   
    
    #-------------training---------------------
    training(model_fun1, train_x, train_y)
  
    
    #--------------test---------------------------------
    predict()
    