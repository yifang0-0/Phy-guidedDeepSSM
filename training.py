import os
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import numpy as np
import time
import matplotlib as plt
import utils.datavisualizer as dv
from torch.optim.lr_scheduler import LambdaLR

def run_train(modelstate, loader_train, loader_valid, options, dataframe, path_general, file_name_general):
    def validate(loader):
        modelstate.model.eval()
        total_vloss = 0
        total_points = 0
        with torch.no_grad():
            for u, y in loader:
                u = u.to(options['device'])
                y = y.to(options['device'])
                vloss_ = modelstate.model(u, y)
                total_points += np.prod(u.shape)
                total_vloss += vloss_.item()
        return total_vloss / total_points

    def train(epoch):
        modelstate.model.train()
        total_loss = 0
        total_points = 0

        for i, (u, y) in enumerate(loader_train):
            u = u.to(options['device'])
            y = y.to(options['device'])

            modelstate.optimizer.zero_grad()
            loss_ = modelstate.model(u, y)
            loss_.backward()
            torch.nn.utils.clip_grad_norm_(modelstate.model.parameters(), max_norm=1.0)
            modelstate.optimizer.step()

            total_points += np.prod(u.shape)
            total_loss += loss_.item()

            loss = total_loss / total_points
            if i % train_options.print_every == 0:
                print(
                    'Train Epoch: [{:5d}/{:5d}], Batch [{:6d}/{:6d} ({:3.0f}%)]\tLearning rate: {:.2e}\tLoss: {:.3f}'.format(
                        epoch, train_options.n_epochs, (i + 1), len(loader_train),
                        100. * (i + 1) / len(loader_train),
                        modelstate.optimizer.param_groups[0]['lr'], loss))
        return loss

    try:
        model_options = options['model_options']
        train_options = options['train_options']

        modelstate.model.train()
        vloss = validate(loader_valid)
        all_losses = []
        all_vlosses = []
        all_losses_train = []
        best_vloss = vloss
        start_time = time.time()

        lr = train_options.init_lr
        best_epoch = 0
        path = path_general + 'model/'
        file_name = file_name_general + '_bestModel.ckpt'
        print("file_name: ", file_name)
        modelstate.save_model(0, vloss, time.time() - start_time, path, file_name)

        # Setup scheduler
        def lr_lambda(step):
            warmup_steps = 400
            if step < warmup_steps:
                return step / warmup_steps
            return (warmup_steps ** 0.5) / (step ** 0.5)

        # scheduler = LambdaLR(modelstate.optimizer, lr_lambda)
        patience = 20  # 5 validation tests without improvement
        epochs_no_improve = 0  # place this at the top before the loop
        for epoch in range(0, train_options.n_epochs + 1):
            loss = train(epoch)
            all_losses_train.append(loss)

            if epoch % train_options.test_every == 0:
                vloss = validate(loader_valid)
                all_losses.append(loss)
                all_vlosses.append(vloss)

                # scheduler.step()


                if vloss < best_vloss:
                    best_vloss = vloss
                    modelstate.save_model(epoch, vloss, time.time() - start_time, path, file_name)
                    best_epoch = epoch
                    epochs_no_improve = 0  # reset counter if improved
                else:
                    epochs_no_improve += 1
                print('Train Epoch: [{:5d}/{:5d}], Batch [{:6d}/{:6d} ({:3.0f}%)]\tLearning rate: {:.2e}\tLoss: {:.3f}\tVal Loss: {:.3f}'.format(
                    epoch, train_options.n_epochs, len(loader_train), len(loader_train), 100.,
                    modelstate.optimizer.param_groups[0]['lr'], loss, vloss))

                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch}. No improvement in validation loss for {patience} epochs.\n")
                    break
                # if vloss <= best_vloss:
                #     best_vloss = vloss
                #     modelstate.save_model(epoch, vloss, time.time() - start_time, path, file_name)
                #     best_epoch = epoch
                # rmse = torch.sqrt(loss)

                # if modelstate.optimizer.param_groups[0]['lr'] < train_options.min_lr:
                #     print("\nLearning rate below minimum. Stopping early.\n")
                #     break

    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')
        print('-' * 89)

    time_el = time.time() - start_time

    train_dict = {
        'all_losses': all_losses,
        'all_losses_train': all_losses_train,
        'all_vlosses': all_vlosses,
        'best_epoch': best_epoch,
        'total_epoch': epoch,
        'train_time': time_el
    }

    dv.plot_losscurve(train_dict, options, path_general, file_name_general)

    return train_dict
