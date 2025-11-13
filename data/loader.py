from data.base import DataLoaderExt
# from data.cascaded_tank import create_cascadedtank_datasets
# from data.f16gvt import create_f16gvt_datasets
from data.toy_lgssm import create_toy_lgssm_datasets
from data.toy_lgssm_5_pre import create_toy_lgssm_5_datasets
from data.toy_lgssm_2dy_5_pre import create_toy_lgssm_2dy_5_datasets
from data.f16gvt import create_f16gvt_datasets
from data.IndustRobo import create_industrobo_datasets
from data.cascTank import create_cascadedtank_datasets


def  load_dataset(dataset, dataset_options, train_batch_size, test_batch_size, **kwargs):

    if dataset == 'cascaded_tank':
        dataset_train, dataset_valid, dataset_test = create_cascadedtank_datasets(dataset_options.seq_len_train,
                                                                                  dataset_options.seq_len_val,
                                                                                  dataset_options.seq_len_test,**kwargs)
        # Dataloader
        loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
        loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
        loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)
        

    elif dataset == 'toy_lgssm':
        dataset_train, dataset_valid, dataset_test = create_toy_lgssm_datasets(dataset_options.seq_len_train,
                                                                               dataset_options.seq_len_val,
                                                                               dataset_options.seq_len_test,
                                                                               **kwargs)
        # Dataloader
        loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
        loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
        loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)
    
    elif dataset == 'toy_lgssm_5_pre':
        dataset_train, dataset_valid, dataset_test = create_toy_lgssm_5_datasets(dataset_options.seq_len_train,
                                                                               dataset_options.seq_len_val,
                                                                               dataset_options.seq_len_test,
                                                                               **kwargs)
        # Dataloader
        loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
        loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
        loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)
        
    # elif dataset == 'toy_lgssm_2dy_5_pre':
    #     dataset_train, dataset_valid, dataset_test = create_toy_lgssm_2dy_5_datasets(dataset_options.seq_len_train,
    #                                                                            dataset_options.seq_len_val,
    #                                                                            dataset_options.seq_len_test,
    #                                                                            **kwargs)
    #     # Dataloader
    #     loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
    #     loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
    #     loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)

        
    # elif dataset == 'f16gvt':
    #     dataset_train, dataset_valid, dataset_test = create_f16gvt_datasets(dataset_options.seq_len_train,
    #                                                                         dataset_options.seq_len_val,
    #                                                                         dataset_options.seq_len_test,
    #                                                                         dataset_options.input_type,
    #                                                                         dataset_options.input_lev,
    #                                                                         **kwargs
    #                                                                         )
    #     # Dataloader
    #     loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
    #     loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
    #     loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)


    elif dataset == 'industrobo':
        print("dataset_options.seq_stride," ,dataset_options.seq_stride)
        dataset_train, dataset_valid, dataset_test = create_industrobo_datasets(seq_len_train = dataset_options.seq_len_train,
                                                                            seq_len_val = dataset_options.seq_len_val,
                                                                            seq_len_test = dataset_options.seq_len_test,
                                                                            seq_stride = dataset_options.seq_stride,
                                                                            sample_rate = dataset_options.dt,
                                                                            input_lev = dataset_options.input_channel,
                                                                            file_name = dataset_options.if_simulation,
                                                                            **kwargs
                                                                            )
        # Dataloader
        loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
        loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
        loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)
    else:
        raise Exception("Dataset not implemented: {}".format(dataset))

    return {"train": loader_train, "valid": loader_valid, "test": loader_test}
