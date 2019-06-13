# -*- coding: utf-8 -*-
from io import open
import numpy as np
import time

# save loss (for plot)
def save_loss(plot_epoches, plot_losses, plot_val_losses, dataset, learning_rate, batch_size, checkpoint):
    dic = {'plot_epoches': plot_epoches, 'plot_losses': plot_losses, 'plot_val_losses': plot_val_losses,
           'dataset': dataset, 'learning_rate': learning_rate, 'batch_size': batch_size, 'checkpoint': checkpoint}
    np.save('loss/' + 'loss' + '.npy', dic)  # 每次重写会覆盖


# write loss log
def write_log_head(train_param):
    t = time.strftime("%m-%d %H:%M", time.localtime())
    with open('loss/loss_log', 'a') as f:
        f.write('\n\n')
        f.write(t)
        f.write('\n' + str(train_param) + '\n')

# def write_log_head(dataset, learning_rate, batch_size, checkpoint):
#     t = time.strftime("%m-%d %H:%M", time.localtime())
#     with open('loss/loss_log', 'a') as f:
#         f.write('\n{0} \ndataset = {1} \nlearning rate = {2} \nbatch size = {3} \ncheckpoint = {4} \n'
#                 .format(t, dataset, str(learning_rate), str(batch_size), checkpoint))


def write_log_loss(epoch, plot_loss_avg, val_loss_avg):
    with open('loss/loss_log', 'a') as f:
        f.write(
            'epoch: {0} | train loss: {1} | val loss: {2}\n'.format(str(epoch), str(plot_loss_avg), str(val_loss_avg)))
