# import generic libraries
"""
The `run_main_single` function is the main function that runs the training and testing of a single
model on a given dataset.

:param options: - dataset: The name of the dataset to use. Options are 'narendra_li', 'toy_lgssm',
'wiener_hammerstein'
:param path_general: The `path_general` variable is the path where the log files and data files will
be saved. It is a combination of the current working directory, the log directory specified in the
options, the dataset name, and the model name
:param file_name_general: The `file_name_general` parameter is a string that specifies the general
file name for saving the results of the experiment. It is used to create a unique file name for each
experiment by appending additional information such as the model type, dynamic system type, and
model parameters (h_dim, z_dim,
"""

import torch.utils.data
import pandas as pd
import os
import torch
import time
import subprocess
import io
import json
# os.chdir('../')
# sys.path.append(os.getcwd())
# import user-written files
import data.loader as loader
import training
import testing
from utils.utils import compute_normalizer
from utils.logger import set_redirects
from utils.utils import save_options
# import options files
import options.train_options as main_params
import options.train_options as model_params
import options.train_options as dynsys_params
import options.train_options as train_params
from models.model_state import ModelState


# %%####################################################################################################################
# Main function
########################################################################################################################
def run_main_single(options, path_general, file_name_general):
    start_time = time.time()
    print(time.strftime("%c"))
    
   # get correct computing device

    if torch.cuda.is_available():
        
        # get the usage of gpu memory from command "nvidia-smi" 
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        gpu_df = pd.read_csv(io.BytesIO(gpu_stats),names=['memory.used', 'memory.free'],skiprows=1)
        print('GPU usage:\n{}'.format(gpu_df))
        
        #get the id of the gpu with a maximum memory space left
        gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
        idx = gpu_df['memory.free'].astype(int).idxmax()        
        print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
        
        # run the task on the selected GPU
        torch.cuda.set_device(idx)
        if int(gpu_df.iloc[idx]['memory.free'])<1000:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda') 
            gpu_name = torch.cuda.get_device_name(idx)
            print(f"Using GPU {idx}: {gpu_name}") 
    else:
        device = torch.device('cpu')
    print('Device: {}'.format(device))



    # get the options
    options['device'] = device
    options['dataset_options'] = dynsys_params.get_dataset_options(options['dataset'])
    options['train_options'] = train_params.get_train_options(options['dataset'])
    options['system_options'] = dynsys_params.get_system_options(options['dataset'],options['dataset_options'],options['train_options'])
    options['model_options'] = model_params.get_model_options(options['model'], options['dataset'],
                                                              options['dataset_options'])
    options['test_options'] = train_params.get_test_options()

    # print model type and dynamic system type
    print('\n\tModel Type: {}'.format(options['model']))
    print('\tDynamic System: {}\n'.format(options['dataset']))

    if options["dataset"] == "f16gvt":
        file_name_general = file_name_general + '_h{}_z{}_n{}_{}'.format(options['model_options'].h_dim,
                                                                  options['model_options'].z_dim,
                                                                  options['model_options'].n_layers,
                                                                  options['dataset_options'].input_type + str(options['dataset_options'].input_lev))
    else:
        file_name_general = file_name_general + '_h{}_z{}_n{}'.format(options['model_options'].h_dim,
                                                                  options['model_options'].z_dim,
                                                                 options['model_options'].n_layers)
        
    if "mpnt_wt" in options["model_options"]:
        file_name_general = file_name_general + '_mpw'+str( int(options["model_options"].mpnt_wt*10))
        
    if "A_prt_idx" in options["dataset_options"]:
        file_name_general = file_name_general + '_A'+str(options["dataset_options"].A_prt_idx)
        
    if "B_prt_idx" in options["dataset_options"]:
        file_name_general = file_name_general + '_B'+str(options["dataset_options"].B_prt_idx)
    
    if "C_prt_idx" in options["dataset_options"]:
        print("C here!")
        file_name_general = file_name_general + '_C'+str(options["dataset_options"].C_prt_idx)

    path = path_general + 'data/'
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # set logger
    set_redirects(path, file_name_general)


    # save the options
    save_options(options, path_general, 'options.txt')

    # initialize dataset
    train_rounds = options["train_rounds"]
    start_from_round = options["start_from"]
    # print number of evaluations
    print('Total number of data point sets: {}'.format(train_rounds))

    # allocation
    all_vaf = torch.zeros([train_rounds])
    all_rmse = torch.zeros([train_rounds])
    all_nrmse = torch.zeros([train_rounds])
    all_likelihood = torch.zeros([train_rounds])
    all_df = {}

    # initialize the dataframe
    
    for i in range(start_from_round, train_rounds):

        file_name_general_i=file_name_general+"_No"+str(i)
        torch.cuda.empty_cache()

        # print which training iteration this is
        print(' {}/{} round starts'.format(i+1,train_rounds))
            # Specifying datasets
        loaders = loader.load_dataset(dataset=options["dataset"],
                                  dataset_options=options["dataset_options"],
                                  train_batch_size=options["train_options"].batch_size,
                                  test_batch_size=options["test_options"].batch_size, 
                                  known_parameter=options["known_parameter"],
                                  k_max_train = options["dataset_options"].k_max_train,
                                  k_max_test = options["dataset_options"].k_max_test,
                                  k_max_val = options["dataset_options"].k_max_val,
                                  train_rounds = train_rounds,
                                  ith_round = i)
        
        # Compute normalizers
        if options["normalize"]:
            print("normalized")
            normalizer_input, normalizer_output = compute_normalizer(loaders['train'])
            print(" normalizer_input.scale, normalizer_input.offset,normalizer_output.scale, normalizer_output.offset: ", normalizer_input.scale, normalizer_input.offset,normalizer_output.scale, normalizer_output.offset)
            
        else:
            print("no normalized")
            normalizer_input = normalizer_output = None
        # python main_single.py --n_epoch=1000 --model='AE-TRANSFORMER' --h_dim=64  --dataset='cascaded_tank' --logdir='0417_notnormalize_shift'  --mpnt_wt=-1   --do_test  --z_dim=2 --u_dim=1 --y_dim=1 --init_lr=1e-3     --do_train --normalize
        # Define model
        # for real dataset use different random seed
        modelstate = ModelState(seed=options["seed"]+i*10,
                                nu=loaders["train"].nu, ny=loaders["train"].ny,
                                model=options["model"],
                                options=options,
                                normalizer_input=normalizer_input,
                                normalizer_output=normalizer_output,
                                )
        modelstate.model.to(options['device'])
        if options['do_train']:
            # train the model

            print("\n\n yes training started!\n\n")
            try:
                # print("options['device']",options['device'])
                new_df = []
                new_df = training.run_train(modelstate=modelstate,
                                        loader_train=loaders['train'],
                                        loader_valid=loaders['valid'],
                                        options=options,
                                        dataframe=new_df,
                                        path_general=path_general,
                                        file_name_general=file_name_general_i)
            
                # check if path exists and create otherwise
                print(new_df)
                # df_list = []
                # for _,i_df in df.items():
                #     print(i_df)
                #     df_list.append(i_df)
                # df = pd.concat(df_list)
                with open(path_general+file_name_general_i + '_trainingrecord.json', 'w') as f:
                    json.dump(new_df, f)
            except Exception as e:
                print(f" Training failed for round {i+1}, skipping to next round.")
                print("Error:", e)
                continue
            # new_df.to_csv(path_general+file_name_general_i + '_trainingrecord.csv',index=False)

        
        if options['do_test']:
            # # test the model
            try:
                # make sure df is in dataframe format
                df = pd.DataFrame({})
                # for i in range(10):
                df_single = {}
                # test the model
                df_single = testing.run_test(options, loaders, df, path_general, file_name_general_i)
                # make df_single a dataframe
                df_single = pd.DataFrame(df_single)
                df = pd.concat([df,df_single])

                if options['savelog']:
                    df.to_csv(path + file_name_general_i,index=False)

                # store values
                all_df[i] = df

                # save performance values
                # print(df['vaf'],df['vaf'][0],type(df['rmse']),type(df['vaf']))
                all_vaf[i] = df['vaf'][0]
                all_rmse[i] = df['rmse'][0]
                all_nrmse[i] = df['nrmse'][0]
            except Exception as e:
                print(f"Testing failed for round {i+1}, skipping to next round.")
                print("Error:", e)
            # all_likelihood[i] = df['marginal_likeli'].item() 
        
    # %% save data
        # save data
    if options['savelog']==True and options['do_test']==True:
        # df.to_csv(path + file_name,index=False)
        # # get saving path
        # path = path_general + 'data/'
        # check if path exists and create otherwise
        if not os.path.exists(path):
            os.makedirs(path)
        # to pandas
        all_df_list = []
        for _,i_df in all_df.items():
            all_df_list.append(i_df)
        all_df = pd.concat(all_df_list)
            
        print(all_df)


        file_name = file_name_general + '_multitrain.csv'
        
        # check if path exists and create otherwise
        if not os.path.exists(path):
            os.makedirs(path)
        # # check if there's old log
        if os.path.exists(path + file_name):
            # read old data
            df_old = pd.read_csv(path + file_name,index_col=None)
            #append new to the old file
            df = df_old.append(df)
        # save data
        all_df.to_csv(path_general + file_name)
        # save performance values
        torch.save(all_vaf, path_general + 'data/' + 'all_vaf.pt')
        torch.save(all_rmse, path_general + 'data/' + 'all_rmse.pt')
        torch.save(all_likelihood, path_general + 'data/' + 'all_likelihood.pt')
        
        
        



    # time output
    time_el = time.time() - start_time
    hours = time_el // 3600
    min = time_el // 60 - hours * 60
    sec = time_el - min * 60 - hours * 3600
    print('Total ime of file execution: {}:{:2.0f}:{:2.0f} [h:min:sec]'.format(hours, min, sec))
    print(time.strftime("%c"))


# %%
# The `if __name__ == "__main__":` block is used to check if the current script is being run as the
# main program. If it is, then the code inside the block will be executed.

if __name__ == "__main__":
    # set (high level) options dictionary, if the basic options are expected from the augment parser, we set OPTION_SETTING_MANUALLY = True, else we change the options directly from the python file.
    OPTION_FROM_PARSER = True
    if OPTION_FROM_PARSER is True:
        options = {}
        main_params_parser = main_params.get_main_options()
        options['dataset'] = main_params_parser.dataset
        options['model'] = main_params_parser.model
        options['do_train'] = main_params_parser.do_train
        options['do_test'] = main_params_parser.do_test
        options['logdir'] = main_params_parser.logdir
        options['normalize'] = main_params_parser.normalize
        options['seed'] = main_params_parser.seed
        options['optim'] = main_params_parser.optim
        options['showfig'] = main_params_parser.showfig
        options['savefig'] = main_params_parser.savefig
        options['savelog'] = main_params_parser.savelog
        options['saveoutput'] = main_params_parser.saveoutput
        options['known_parameter'] = main_params_parser.known_parameter
        options['train_rounds'] = main_params_parser.train_rounds
        options['start_from'] = main_params_parser.start_from

        # print("Encountered errors loading the main options of the training/testing task")
        print("normalizer: ",options['normalize'])
        
    else:
        options = {
            'dataset': 'toy_lgssm',  # options: 'f16gvt', 'narendra_li', 'toy_lgssm', 'wiener_hammerstein', 'industrobo','toy_lgssm_5_pre'
            'model': 'VAE-RNN-PHYNN', # options: 'VAE-RNN', 'VRNN-Gauss', 'VRNN-Gauss-I', 'VRNN-GMM', 'VRNN-GMM-I', 'STORN', 'VAE-RNN-PHYNN'
            'do_train': True,
            'do_test': True,
            'logdir': 'high_panelty', 
            'normalize': True,
            'seed': 1234,
            'optim': 'Adam',
            'showfig': True,
            'savefig': True,
            'savelog': False,
            'saveoutput':True,
            'known_parameter': 'None'
        }

    # get saving path
    
    path_general = os.getcwd() + '/log/{}/{}/{}_{}/'.format(options['logdir'],
                                                         options['dataset'],
                                                         options['model'],options['known_parameter'] )
        

    # get saving file names
    file_name_general = options['dataset']

    run_main_single(options, path_general, file_name_general)
