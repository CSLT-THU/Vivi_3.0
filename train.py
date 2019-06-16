# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import random
import os
import time
import math
import numpy as np
import argparse
import numpy as np
import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset

import configparser
import importlib

from data_utils import read_train_data, read_train_data_2, read_test_data, split_dataset, sort_batch_data
# from loss.loss_logs import save_loss, write_log_head, write_log_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def val(val_data):
    return 0


def train_batch(batch_size, data, 
                model, optimizer, criterion, teacher_forcing_ratio): 
    
    optimizer.zero_grad()
    
    loss = model(batch_size, data, criterion,
                teacher_forcing_ratio)
    
    loss.backward()
    
    optimizer.step()

    return loss.item() # 平均一个句子的loss
    # return loss.item() / int(sum(target_lengths)) 
    # 问题：除多了，loss本身就对batch的每一个sequence取过平均了，而这个sum包含batch中的80个句子。（target_length是一个batch的）
    # 所以一共除了batch size * seq len。 如果想取字的平均，应只除80个句子的平均seq len，不应除batch size。


def train(train_data, val_data, model, optimizer, batch_size, epochs, last_epoch, val_rate, teacher_forcing_ratio, model_param, train_param):
    model.train()

    # plot loss init
    plot_losses = []
    plot_epoches = []
    plot_val_losses = []

    # criterion
    criterion = nn.CrossEntropyLoss()  # 对batch取平均

    # steps = len(train_set) // batch_size # how many steps(batch) in an epoch 取整数 抛弃不能整除的部分数据

    for i in range(epochs):
        epoch = i + last_epoch + 1  # epoch从1开始
        print('epoch: %d' % epoch)

        step = 0
        loss_total = 0
        start = time.time()

        # 进度条 有bug
        # for batch in tqdm(train_data, mininterval=2, desc='  - Training ', leave=False):
        #     src_seq, src_pos, tgt_seq, tgt_pos = batch

        for step, data in enumerate(train_data):
            # 每一步 loader 释放一小批数据用来学习，step=总数据量/batch_size，enumerate把每次提取编写索引。
            # batch_x: B*T tensor
            
            # 最后一个batch数据量可能不足batch size，去掉不用。
            # if len(data[0]) != batch_size:
            #     continue
            
            step += 1
            if step % 100 == 0: # 临时
                print('step:', step)

            loss = train_batch(batch_size, data, model, optimizer, criterion, teacher_forcing_ratio)
            loss_total += loss

        loss_avg = round((loss_total / step), 5)
        print(' - Training loss: {loss:}(per sentence), elapse: {elapse:3.1f} min'.format(
            loss=loss_avg, elapse=(time.time() - start) / 60))

        # validation 
        val_loss_avg = 0
        if val_rate:
            val_loss_avg = val(val_data)  # for the whole val set 
            print(' - Validation loss: %.1f' % val_loss_avg)
            plot_val_losses.append(val_loss_avg)

        # write loss log, save loss for every epoch, in case of interruption
        plot_losses.append(loss_avg)
        plot_epoches.append(epoch)
        with open('loss/loss_log', 'a') as f:
            f.write('epoch: {0} | train loss: {1} | val loss: {2}\n'.format(str(epoch), str(loss_avg), str(val_loss_avg)))
        dic = {'plot_epoches': plot_epoches, 'plot_losses': plot_losses, 'plot_val_losses': plot_val_losses, 'model_param': model_param, 'train_param': train_param}
        np.save('loss/loss.npy', dic)  # 每次重写会覆盖

        # save model for every epoch
        print('save model')
        t = time.strftime("%m-%d", time.localtime())  # "%m-%d-%H:%M"
        state = {'model': model.state_dict(), 'train_param':train_param, 'model_param': model_param, 'epoch': epoch}
        torch.save(state, 'ckpt/' + str(t) + '_' + model_param['model_name'] + '_epoch=' + str(epoch) + '_loss=%.1f' % loss_avg + '.pkl')


def main():
    t = time.strftime("%m-%d %H:%M", time.localtime())
    print('\n', t, '\n')
    print('device:', device)

    # ========= Get Parameter =========#
    # train parameters
    parser = argparse.ArgumentParser(description='Vivi')
    
    parser.add_argument('--dataset', type=str, default='resource/dataset/poem_480.txt')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--val_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.8)
    parser.add_argument('--model_name', type=str, default='Transformer')
    parser.add_argument('--note', type=str, default='')
    
    args = parser.parse_args()
    
    dataset = args.dataset
    epochs = args.epochs
    ckpt_path = args.ckpt_path
    val_rate = args.val_rate
    batch_size = args.batch_size
    teacher_forcing_ratio = args.teacher_forcing_ratio
    model_name = args.model_name
    
    train_param = vars(args)
    print('train param: ', train_param)
      
    # load model parameters    
    checkpoint = None
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model_param = checkpoint['model_param']
        last_epoch = checkpoint['epoch']
    else:
        conf = configparser.ConfigParser()
        conf.read('config/config_'+model_name+'.ini')
        model_param_li = conf.items('model_param')
        model_param = {'model_name': model_name}
        for item in model_param_li:
            model_param[item[0]] = item[1]
        last_epoch = 0
    print('model param: ', model_param)

    # ========= Preparing Data =========#

    # read data
    pairs = read_train_data(dataset)
    # split dataset
    train_pairs, val_pairs = split_dataset(pairs, val_rate)  # pairs

    data_path = 'models.' + model_name + '.PoetryData'
    PoetryData = importlib.import_module(data_path)
    train_Dataset = getattr(PoetryData, 'PoetryData')(train_pairs, src_max_len=int(model_param['input_max_len']), 
                                                     tgt_max_len=int(model_param['target_max_len']))
    val_Dataset = getattr(PoetryData, 'PoetryData')(val_pairs, src_max_len=int(model_param['input_max_len']), 
                                                     tgt_max_len=int(model_param['target_max_len']))# 反射并实例化
    
    # 变成小批
    train_data = Data.DataLoader(
        dataset=train_Dataset,  
        batch_size=batch_size, 
        shuffle=True,  
        drop_last=True, # Jun16
        collate_fn=PoetryData.paired_collate_fn # 目前仅适用于transformer
        # num_workers=2,  # 多线程来读数据，提取xy的时候几个数据一起提取
    )
    val_data = Data.DataLoader(
        dataset=val_Dataset,  
        batch_size=batch_size,  
        shuffle=True,
        drop_last=True, # Jun16
        collate_fn=PoetryData.paired_collate_fn
        # num_workers=2,  
    )

    # ========= Preparing Model =========#

    model_path = 'models.' + model_name + '.' + model_name
    Model = importlib.import_module(model_path)  # 导入模块
    model = getattr(Model, model_name)(model_param)  # 反射并实例化
    print('model:', model)
    
    optim_path = 'models.' + model_name + '.Optim'
    Optim = importlib.import_module(optim_path) # 模块（文件）
    optimizer = Optim.get_optimizer(model, model_param) # 调用模块的函数
    print('optimizer:', optimizer)

    # load model from ckpt
    if os.path.exists(ckpt_path):
        # checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage) # 重复load
        model.load_state_dict(checkpoint['model'])

    # write log head
    with open('loss/loss_log', 'a') as f:
        f.write('\n\n' + str(t) + '\n' + str(train_param) + '\n' + str(model_param) + '\n')
        
    print('start training')
    train(train_data, val_data, model, optimizer, batch_size, epochs, last_epoch, val_rate, teacher_forcing_ratio, model_param, train_param)


if __name__ == '__main__':
    main()