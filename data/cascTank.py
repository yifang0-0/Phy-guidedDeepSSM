import matplotlib.pyplot as plt
import numpy as np
from data.base import IODataset
import nonlinear_benchmarks

def create_cascadedtank_datasets(seq_len_train=None, seq_len_val=None, seq_len_test=None, **kwargs):
    print("kwargs:", kwargs)
    # length of all data sets
    if bool(kwargs) and ("k_max_train" in kwargs):
        k_max_train = kwargs['k_max_train']
        k_max_val = kwargs['k_max_val']
        k_max_test = kwargs['k_max_test']
    else:
        # Default option
        k_max_train = 768
        k_max_val = 256
        k_max_test = 1024
    
    print("k_max_train: ", k_max_train, "\nk_max_val: ", k_max_val, "\nk_max_test: ", k_max_test)

    train_val, test = nonlinear_benchmarks.Cascaded_Tanks()
    
    print(test.state_initialization_window_length) # = 50
    train_val_u, train_val_y = train_val
    test_u, test_y = test
    print(train_val_u.shape)
    print(test_u.shape)
    
    
    u_test = test_u[0:k_max_test]
    y_test = test_y[0:k_max_test]
    print("TEST: X[0]", test_y[0])
    print("TRAIN: X[0]", train_val_y[0])
    print("VAL: X[0]",train_val_y[k_max_train-1])
    
    

    u_train = train_val_u[0:k_max_train]
    y_train = train_val_y[0:k_max_train]
    # u_val = train_val_u[k_max_train-k_max_val:k_max_train]
    # y_val = train_val_y[k_max_train-k_max_val:k_max_train]
    train_val_len = len(train_val_u)
    u_val = train_val_u[train_val_len-k_max_val:]
    y_val = train_val_y[train_val_len-k_max_val:]
    # u_val = test_u[0:k_max_val]
    # y_val = test_y[0:k_max_val]
    
    print("dataset: Cascaded_Tanks")

    dataset_train = IODataset(u_train, y_train, seq_len_train, stride=None)
    dataset_val = IODataset(u_val, y_val, seq_len_val, stride=None)
    # dataset_test = IODataset(u_train, y_train, seq_len_train)
    dataset_test = IODataset(u_test, y_test, seq_len_test)
    # dataset_test = IODataset(u_val, y_val, seq_len_val, stride=None)
    # dataset_test = IODataset(u_train, y_train, seq_len_train, stride=None)
    
    
    

    print("dataset_train.u.shape, dataset_train.y.shape")
    print(dataset_train.u.shape, dataset_train.y.shape)
    print("dataset_val.u.shape, dataset_val.y.shape")
    print(dataset_val.u.shape, dataset_val.y.shape)
    print("dataset_test.u.shape, dataset_test.y.shape")
    print(dataset_test.u.shape, dataset_test.y.shape)
    # return dataset_val,dataset_train, dataset_test
    return dataset_train, dataset_val, dataset_test
