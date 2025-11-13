import argparse
import torch
import numpy as np

'''
parse_known_args return the new parser and the already know data
'''

def get_train_options(dataset_name):
    train_parser = argparse.ArgumentParser(description='training parameter')
    train_parser.add_argument('--clip', type=int, default=10, help='clipping of gradients')
    train_parser.add_argument('--lr_scheduler_nstart', type=int, default=10, help='learning rate scheduler start epoch')
    train_parser.add_argument('--print_every', type=int, default=1, help='output print of training')
    train_parser.add_argument('--test_every', type=int, default=5, help='test during training after every n epoch')
    train_parser.add_argument('--known_x0', type=int, default=0, help='initial x0')
    
    
    if dataset_name == 'cascaded_tank':
        train_parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-9, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=10, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='adapt learning rate by')

    elif dataset_name == 'toy_lgssm':
        train_parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-7, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=30, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=5, help='adapt learning rate by')
        # train_parser.add_argument('--unknown_parameter', type=int, default=1, help='0 is the normal training, 1 means linear matrix B is known')
    
    elif dataset_name == 'toy_lgssm_5_pre':
        train_parser.add_argument('--n_epochs', type=int, default=750, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-7, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=30, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=5, help='adapt learning rate by')
        
    elif dataset_name == 'toy_lgssm_2dy_5_pre':
        train_parser.add_argument('--n_epochs', type=int, default=750, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-7, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=30, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=5, help='adapt learning rate by')


    # elif dataset_name == 'wiener_hammerstein':
    #     train_parser.add_argument('--n_epochs', type=int, default=750, help='number of epochs')
    #     train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
    #     train_parser.add_argument('--min_lr', type=float, default=1e-6, help='minimal learning rate')
    #     train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=20, help='check learning rater after')
    #     train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='adapt learning rate by')
        
    # elif dataset_name == 'f16gvt':
    #     train_parser.add_argument('--n_epochs', type=int, default=750, help='number of epochs')
    #     train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
    #     train_parser.add_argument('--min_lr', type=float, default=1e-6, help='minimal learning rate')
    #     train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=20, help='check learning rater after')
    #     train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='adapt learning rate by')
        
    elif dataset_name == 'industrobo':
        train_parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-7, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=20, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='adapt learning rate by')

    # change batch size to higher value if trained on cuda device
    if torch.cuda.is_available():
        train_parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    else:
        train_parser.add_argument('--batch_size', type=int, default=128, help='batch size')


    train_options, unknown = train_parser.parse_known_args()

    return train_options


def get_test_options():
    test_parser = argparse.ArgumentParser(description='testing parameter')
    test_parser.add_argument('--batch_size', type=int, default=32, help='batch size')  # 128
    test_options,unkonwn = test_parser.parse_known_args()

    return test_options


def get_main_options():

    # model parameters
    model_parser = argparse.ArgumentParser(description='Model Parameter')
    model_parser.add_argument('--dataset',metavar='', type=str, default='cascaded_tank')
    model_parser.add_argument('--model', metavar = '', type=str, default='AE-RNN')
    model_parser.add_argument('--do_train', action="store_true")
    model_parser.add_argument('--do_test', action="store_true")
    model_parser.add_argument('--logdir',metavar = '',  type=str, default='same_dataset')
    model_parser.add_argument('--normalize', action='store_false',  default=True, help='if disable normalize, include it')
    model_parser.add_argument('--seed', metavar = '', type=int, default=1234)
    model_parser.add_argument('--optim', metavar = '', type=str, default='Adam')
    model_parser.add_argument('--showfig', metavar = '', type=bool, default=True)
    model_parser.add_argument('--savefig', metavar = '', type=bool, default=True)
    model_parser.add_argument('--savelog', metavar = '', type=bool, default=True)
    model_parser.add_argument('--saveoutput', metavar = '', type=bool, default=True)
    model_parser.add_argument('--known_parameter', metavar = '', type=str, default='None')
    model_parser.add_argument('--train_rounds', metavar = '', type=int, default=1)
    model_parser.add_argument('--start_from', metavar = '', type=int, default=0)
    
    model_options, unknown = model_parser.parse_known_args()
    print(model_options)
    return model_options


def get_system_options(dataset_name,dataset_options, train_options):
    if dataset_name == 'toy_lgssm_5_pre' or dataset_name == 'toy_lgssm':
        system_parameter = {}
        if train_options.known_x0 == 1:
            system_parameter['x0'] = np.array([[0], [0]])
        else:
            system_parameter['x0'] = np.array([[0], [0]])
        system_parameter['A'] = np.array([[0.7, 0.8], [0, 0.1]])
        system_parameter['B'] = np.array([[-1], [0.1]])
        if dataset_options.A_prt_idx==0:
            system_parameter['A_prt'] = np.array([[0, 0], [0, 0]]) 
        elif dataset_options.A_prt_idx==1:
            system_parameter['A_prt'] = np.array([[0.6, 0.7], [-0.1, 0]])
        elif dataset_options.A_prt_idx==2:
            system_parameter['A_prt'] = np.array([[0.7, 0.8], [0, 0.1]])  
            
        if dataset_options.B_prt_idx==0:
            system_parameter['B_prt'] = np.array([[0], [0]]) 
        elif dataset_options.B_prt_idx==1:
            system_parameter['B_prt'] = np.array([[-1.1], [0]])
        elif dataset_options.B_prt_idx==2:
            system_parameter['B_prt'] = np.array([[-1], [0.1]])
    
        if dataset_options.C_prt_idx==0:
            system_parameter['C_prt'] = np.array([[0, 0]]) 
        elif dataset_options.C_prt_idx==1:
            system_parameter['C_prt'] = np.array([[0.9, -0.1]]) 
        elif dataset_options.C_prt_idx==2:
            system_parameter['C_prt'] = np.array([[1, 0]]) 
        # lgssm_system_parameter['C'] = np.array([[1, 0]])
        system_parameter['sigma_state'] = np.sqrt(0.25)
        system_parameter['sigma_out'] = np.sqrt(1)
    
    elif dataset_name == 'toy_lgssm_2dy_5_pre':
        system_parameter = {}
        if train_options.known_x0 == 1:
            system_parameter['x0'] = np.array([[0], [0]])
        else:
            system_parameter['x0'] = np.array([[0], [0]])
        system_parameter['A'] = np.array([[0.7, 0.8], [0, 0.1]])
        system_parameter['B'] = np.array([[-1], [0.1]])
        system_parameter['C'] = np.array([[1, 0], [0, 1]]).transpose()
        system_parameter['sigma_state'] = np.sqrt(0.25)
        system_parameter['sigma_out'] = np.sqrt(1)
        
    elif dataset_name == "industrobo":
        system_parameter = {}
        system_parameter['dt'] = dataset_options.dt
        system_parameter['roboname'] = dataset_options.roboname
        if dataset_options.if_G ==1:
            system_parameter['if_G'] = True
        else:
            system_parameter['if_G'] = False
            
        if dataset_options.if_clip ==1:
            system_parameter['if_clip'] = True
        else:
            system_parameter['if_clip'] = False
        if dataset_options.if_level2 ==1:
            system_parameter['if_level2'] = True
        else:
            system_parameter['if_level2'] = False
        if dataset_options.if_bias ==1:
            system_parameter['if_bias'] = True
        else:
            system_parameter['if_bias'] = False
        if dataset_options.if_level0 ==1:
            system_parameter['if_level0'] = True
        else:
            system_parameter['if_level0'] = False
    elif dataset_name == 'cascaded_tank':
        system_parameter = {}
        system_parameter['dt'] = 4
        
        system_parameter['k1'] = np.array(0.0464)
        system_parameter['k2'] = np.array(0.0003)
        system_parameter['k3'] = np.array(0.0412)
        system_parameter['k4'] = np.array(0.0586)
        system_parameter['k5'] = np.array(0.0039)
        system_parameter['k6'] = np.array(0.0146)
        # system_parameter['offset'] = np.array(-0.1401)
        system_parameter['offset'] = np.array(0)
        
        system_parameter['x2Max'] = np.array(10)
        system_parameter['xMin'] = np.array(0)
        
        system_parameter['x1Max'] = np.array(10)
        
        
        if train_options.known_x0 == 1:
            system_parameter['x0'] = np.array([[3.369], [3.369]])
        else:
            system_parameter['x0'] = np.array([[3.369], [3.369]])
# TEST: X[0] 4.972762645914397
# TRAIN: X[0] 5.205004959182117
# VAL: X[0] 3.369039444571603
        system_parameter['dt'] = dataset_options.dt
    else:
        system_parameter = {}
    return system_parameter

def get_dataset_options(dataset_name):

    if dataset_name == 'cascaded_tank':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: cascaded tank')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=128, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=1024, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=128, help='validation sequence length')
        dataset_parser.add_argument('--dt', type=float, default=4, help='sampling rate here')
        dataset_parser.add_argument('--k_max_train', type=int, default=128 , help='training set length')
        dataset_parser.add_argument('--k_max_test', type=int, default=1024, help='test set length')
        dataset_parser.add_argument('--k_max_val', type=int, default=128, help='validation set length')  # 512
        dataset_options, unkonwn = dataset_parser.parse_known_args()

    elif dataset_name == 'toy_lgssm':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: lgssm')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=64, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=None, help='test sequence length')
        dataset_parser.add_argument('--loss_type', type=int, default=0, help='0:normal loss, 1:measurement penalty')
        dataset_parser.add_argument('--A_prt_idx', type=int, default=0, help='0:no knowledge, 1:know with bias, 2:the identical A')
        dataset_parser.add_argument('--B_prt_idx', type=int, default=0, help='0:no knowledge, 1:know with bias, 2:the identical B')
        dataset_parser.add_argument('--C_prt_idx', type=int, default=0, help='0:no knowledge, 1:know with bias, 2:the identical C')
        dataset_parser.add_argument('--seq_len_val', type=int, default=64, help='validation sequence length')  # 512
        dataset_options, unknown = dataset_parser.parse_known_args()

    elif dataset_name == 'toy_lgssm_5_pre':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: lgssm')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        # dataset_parser.add_argument('--x_phy_w', type=float, default=1, help='phy weight here')
        # dataset_parser.add_argument('--x_nn_w', type=float, default=1, help='nn weight here') 
        dataset_parser.add_argument('--seq_len_train', type=int, default=64, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=None, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=64, help='validation sequence length')  # 512
        dataset_parser.add_argument('--loss_type', type=int, default=0, help='0:normal loss, 1:measurement penalty')
        dataset_parser.add_argument('--A_prt_idx', type=int, default=0, help='0:no knowledge, 1:know with bias, 2:the identical A')
        dataset_parser.add_argument('--B_prt_idx', type=int, default=0, help='0:no knowledge, 1:know with bias, 2:the identical B')
        dataset_parser.add_argument('--C_prt_idx', type=int, default=0, help='0:no knowledge, 1:know with bias, 2:the identical C')
        dataset_parser.add_argument('--k_max_train', type=int, default=2000, help='training set length')
        dataset_parser.add_argument('--k_max_test', type=int, default=5000, help='test set length')
        dataset_parser.add_argument('--k_max_val', type=int, default=2000, help='validation set length')  # 512
        dataset_options, unknown = dataset_parser.parse_known_args()
    
    elif dataset_name == 'toy_lgssm_2dy_5_pre':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: lgssm')
        dataset_parser.add_argument('--y_dim', type=int, default=2, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=128, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=None, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=128, help='validation sequence length')  # 512
        dataset_parser.add_argument('--loss_type', type=int, default=0, help='0:normal loss, 1:measurement penalty')
        dataset_parser.add_argument('--k_max_train', type=int, default=2000, help='training set length')
        dataset_parser.add_argument('--k_max_test', type=int, default=5000, help='test set length')
        dataset_parser.add_argument('--k_max_val', type=int, default=2000, help='validation set length')  # 512
        dataset_options, unknown = dataset_parser.parse_known_args()
        
    elif dataset_name == 'industrobo':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: industrobo')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=2, help='dimension of u, the input with time')
        dataset_parser.add_argument('--input_channel', type=int, default=1, help='train which joint')
        dataset_parser.add_argument('--dt', type=float, default=0.1, help='sampling rate here')
        dataset_parser.add_argument('--if_clip', type=int, default=1, help='if clip 1, else 0')
        dataset_parser.add_argument('--if_G', type=int, default=1, help='if know Gear info 1, else 0')
        dataset_parser.add_argument('--if_level2', type=int, default=0, help='if knowledge level 2, 1 else 0, default 0')
        dataset_parser.add_argument('--if_bias', type=int, default=0, help='if knowledge level 2, 1 else 0, default 0')
        dataset_parser.add_argument('--if_level0', type=int, default=1, help='if knowledge level 0, 1 else 0, default 1')
        dataset_parser.add_argument('--if_simulation', type=int, default=0, help='if use simulated dataset 1, else 0')
        dataset_parser.add_argument('--roboname', type=str, default="KUKA300", help='choose which robot model we use here')
        # dataset_parser.add_argument('--input_type', type=str, default="FullMSine", help='input activation level')
        dataset_parser.add_argument('--seq_stride', type=int, default=None, help='window size stride')
        dataset_parser.add_argument('--seq_len_train', type=int, default=606, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=606, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=606, help='validation sequence length')
        dataset_parser.add_argument('--k_max_train', type=float, default=1, help='percentage of used length to the full training set length 35990')
        dataset_parser.add_argument('--k_max_test', type=int, default=3636, help='test set length')
        dataset_parser.add_argument('--k_max_val', type=int, default=3998, help='validation set length')  # 512
        dataset_options, unknown = dataset_parser.parse_known_args()


    return dataset_options


def get_model_options(model_type, dataset_name, dataset_options):

    y_dim = dataset_options.y_dim
    u_dim = dataset_options.u_dim

    # model parameters
    model_parser = argparse.ArgumentParser(description='Model Parameter')
    model_parser.add_argument('--y_dim', type=int, default=y_dim, help='dimension of y')
    model_parser.add_argument('--u_dim', type=int, default=u_dim, help='dimension of u')
    model_parser.add_argument('--x_phy_w', type=float, default=1, help='phy weight here')
    model_parser.add_argument('--x_nn_w', type=float, default=1, help='nn weight here') 


    if dataset_name == 'cascaded_tank':
        model_parser.add_argument('--h_dim', type=int, default=60, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=2, help='dimension of stoch. latent variable') 
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')     
        
    elif dataset_name == 'toy_lgssm_5_pre':
        model_parser.add_argument('--h_dim', type=int, default=10, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=2, help='dimension of stoch. latent variable')
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')
        
    elif dataset_name == 'toy_lgssm':
        model_parser.add_argument('--h_dim', type=int, default=70, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=2, help='dimension of stoch. latent variable')
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')
    
    elif dataset_name == 'industrobo':
        model_parser.add_argument('--h_dim', type=int, default=50, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=5, help='dimension of stoch. latent variable')
        model_parser.add_argument('--n_layers', type=int, default=3, help='number of RNN layers (GRU)')


    model_parser.add_argument('--mpnt_wt', type=float, default=0, help='how heavy is the measurement matrix')

    model_options, unkown = model_parser.parse_known_args()

    return model_options