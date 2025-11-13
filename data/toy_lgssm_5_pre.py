import matplotlib.pyplot as plt
import numpy as np
from data.base import IODataset


def run_toy_lgssm_sim(u, A, B, C, sigma_state, sigma_out):
    # just a standard linear gaussian state space model. Measurement Noise is considered outside
    # same system as in toy examples of "Learning of state-space models with highly informative observations: a tempered
    # Sequential Monte Carlo solution", chapter 5.1
    # additionally measurement noise considered

    # get length of input
    k_max = u.shape[-1]

    # size of variables
    n_u = 1
    n_y = 1
    n_x = 2

    # allocation
    x = np.zeros([n_x, k_max + 1])
    y = np.zeros([n_y, k_max])

    # run over all time steps
    for k in range(k_max):
        x[:, k + 1] = np.dot(A, x[:, k]) + np.dot(B, u[:, k]) + sigma_state * np.random.randn(n_x)
        y[:, k] = np.dot(C, x[:, k])

    return y


def create_toy_lgssm_5_datasets(seq_len_train=None, seq_len_val=None, seq_len_test=None, **kwargs):
    print("kwargs:", kwargs)
    # length of all data sets
    if bool(kwargs) and ("k_max_train" in kwargs):
        k_max_train = kwargs['k_max_train']
        k_max_val = kwargs['k_max_val']
        k_max_test = kwargs['k_max_test']
    else:
        # Default option
        k_max_train = 2000
        k_max_val = 2000
        k_max_test = 5000
    
    print("k_max_train: ", k_max_train, "\nk_max_val: ", k_max_val, "\nk_max_test: ", k_max_test)

    # test set input
    file_path_train = 'data/Toy_LGSSM/toy_lgssm_pre_trainingset_shifted_{}.npz'
    file_path_test = 'data/Toy_LGSSM/toy_lgssm_pre_testset_shifted.npz'
    
    ith_lgssm = kwargs["ith_round"]
    
    test_data = np.load(file_path_test)
    u_test = test_data['u_test'][0:k_max_test]
    y_test = test_data['y_test'][0:k_max_test]
    
    train_data = np.load(file_path_train.format(ith_lgssm))
    u_train = train_data['u_train'][0:k_max_train]
    y_train = train_data['y_train'][0:k_max_train]
    u_val = train_data['u_val'][0:k_max_val]
    y_val = train_data['y_val'][0:k_max_val]
    
    print("dataset: ", file_path_train.format(ith_lgssm))
    dataset_train = IODataset(u_train, y_train, seq_len_train)
    dataset_val = IODataset(u_val, y_val, seq_len_val)
    dataset_test = IODataset(u_test, y_test, seq_len_test)
    # # Create datasets with 50% overlap
    # # For training data
    # if seq_len_train is not None:
    #     stride_train = seq_len_train // 2  # 50% overlap
    #     dataset_train = IODataset(u_train, y_train, seq_len_train, stride=stride_train)
    # else:
    #     dataset_train = IODataset(u_train, y_train, seq_len_train)

    # # For validation data
    # if seq_len_val is not None:
    #     stride_val = seq_len_val // 2  # 50% overlap
    #     dataset_val = IODataset(u_val, y_val, seq_len_val, stride=stride_val)
    # else:
    #     dataset_val = IODataset(u_val, y_val, seq_len_val)

    # # For test data
    # if seq_len_test is not None:
    #     stride_test = seq_len_test // 2  # 50% overlap
    #     dataset_test = IODataset(u_test, y_test, seq_len_test, stride=stride_test)
    # else:
    #     dataset_test = IODataset(u_test, y_test, seq_len_test)

    print("dataset_train.u.shape, dataset_train.y.shape")
    print(dataset_train.u.shape, dataset_train.y.shape)
    print("dataset_val.u.shape, dataset_val.y.shape")
    print(dataset_val.u.shape, dataset_val.y.shape)
    print("dataset_test.u.shape, dataset_test.y.shape")
    print(dataset_test.u.shape, dataset_test.y.shape)
    return dataset_train, dataset_val, dataset_test
