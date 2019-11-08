# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import random
import os
import time
import math
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

import data_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
torch.backends.cudnn.deterministic = True

def val(batch_size, data, model, optimizer, criterion, teacher_forcing_ratio, train_param):
    with torch.no_grad():
        loss = model(batch_size, data, criterion, teacher_forcing_ratio, train_param)
    return loss.item()


def train_batch(batch_size, data, model, optimizer, criterion, teacher_forcing_ratio, train_param): 
    optimizer.zero_grad()
    loss = model(batch_size, data, criterion, teacher_forcing_ratio, train_param)
    loss.backward()
    optimizer.step()
    return loss.item() # 平均一个句子的loss
    
    # return loss.item() / int(sum(target_lengths)) 
    # 问题：除多了，loss本身就对batch的每一个sequence取过平均了，而这个sum包含batch中的80个句子。（target_length是一个batch的）
    # 所以一共除了batch size * seq len。 如果想取字的平均，应只除80个句子的平均seq len，不应除batch size。


def train(train_data, val_data, model, optimizer, batch_size, epochs, last_epoch, val_rate, teacher_forcing_ratio, save_dir, t, model_param, train_param):
    model.train()

    # plot loss init
    plot_losses = []
    plot_epoches = []
    plot_val_losses = []

    # criterion
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(reduction='none')  # 取平均 # Jun21

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
            # batch_x: B*T tensor
            
            step += 1
            if step % 100 == 0: # 临时
                print('step:', step)

            loss = train_batch(batch_size, data, model, optimizer, criterion, teacher_forcing_ratio, train_param)
            loss_total += loss

        loss_avg = round((loss_total / step), 5)
        print(' - Training loss: {loss:}(per sentence), elapse: {elapse:3.1f} min'.format(
            loss=loss_avg, elapse=(time.time() - start) / 60))

        # validation 
        if val_rate:
            val_loss_total = 0
            for step, data in enumerate(val_data):
                val_loss = val(batch_size, data, model, optimizer, criterion, teacher_forcing_ratio, train_param)  # for the whole val set 
                val_loss_total += val_loss
            val_loss_avg = round((val_loss_total / step), 5)
            print(' - Validation loss: %.1f' % val_loss_avg)
            plot_val_losses.append(val_loss_avg)

        # write loss log, save loss for every epoch, in case of interruption
        plot_losses.append(loss_avg)
        plot_epoches.append(epoch)
        with open(save_dir+'log', 'a') as f:
            f.write('epoch: {0} | train loss: {1} | val loss: {2}\n'.format(str(epoch), str(loss_avg), str(val_loss_avg)))
        dic = {'plot_epoches': plot_epoches, 'plot_losses': plot_losses, 'plot_val_losses': plot_val_losses, 'model_param': model_param, 'train_param': train_param}
        np.save(save_dir+'loss.npy', dic)  # 每次重写会覆盖
        # np.save('loss/loss.npy', dic)  # 每次重写会覆盖

        # save model for every epoch
        state = {'model': model.state_dict(), 'train_param':train_param, 'model_param': model_param, 'epoch': epoch}
        torch.save(state, save_dir + save_dir.split('/')[1] + '_ep=' + str(epoch) + '_loss=%.2f' % loss_avg + '.pkl')
        print('model saved')
        
def main():
    t = time.localtime()
    t_mark = time.strftime("%m-%d %H:%M", t)
    print('\n', t_mark, '\n')
    print('device:', device)

    # ========= Get Parameter =========#
    # train parameters
    parser = argparse.ArgumentParser(description='Vivi')
    
    parser.add_argument('--dataset', type=str, default='poem_1031k_theme_train')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--val_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.8)
    parser.add_argument('--model_name', type=str, default='Seq2seq_12')
    parser.add_argument('--train_mode', type=str, default='kw2poem') # nL21L or kw2poem
    parser.add_argument('--note', type=str, default='')
    
    parser.add_argument('--train_soft', type=bool, default=True) # Jul12
    parser.add_argument('--template', type=bool, default=False)  # Jul12
    parser.add_argument('--w1', type=float, default=3.)
    parser.add_argument('--w2', type=float, default=0.)

    
    args = parser.parse_args()
    
    dataset = args.dataset
    dataset_path = 'resource/dataset/'+dataset+'.txt'
    epochs = args.epochs
    ckpt_path = args.ckpt_path
    val_rate = args.val_rate
    batch_size = args.batch_size
    teacher_forcing_ratio = args.teacher_forcing_ratio
    model_name = args.model_name
    train_mode = args.train_mode
    
    train_param = vars(args)
      
    # load model parameters    
    checkpoint = None
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model_param = checkpoint['model_param']
        train_param = checkpoint['train_param']
        last_epoch = checkpoint['epoch']
    else:
        conf = configparser.ConfigParser()
        conf.read('config/config_'+model_name+'.ini')
        model_param_li = conf.items('model_param')
        model_param = {'model_name': model_name}
        for item in model_param_li:
            model_param[item[0]] = item[1]
        last_epoch = 0
        
    print('train param: ', train_param)
    print('model param: ', model_param)

    # ========= Preparing Data =========#

    # read data
    if model_name == 'BERT':
        pairs = data_utils.read_BERT_train_data(dataset)
    elif train_mode == 'nL21L':
        pairs = data_utils.read_nL21L_train_data(dataset_path)
    else:
        pairs = data_utils.read_train_data(dataset_path)
    
    # split dataset
    train_pairs, val_pairs = data_utils.split_dataset(pairs, val_rate)  # pairs

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
        collate_fn=PoetryData.paired_collate_fn 
        # num_workers=2  # 多线程来读数据，提取xy的时候几个数据一起提取
    )
    val_data = Data.DataLoader(
        dataset=val_Dataset,  
        batch_size=batch_size,  
        shuffle=True,
        drop_last=True, # Jun16
        collate_fn=PoetryData.paired_collate_fn
        # num_workers=2  
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
    save_dir = 'ckpt/' + str(time.strftime("%m%d%H%M%S", t)) + '_' + model_param['model_name'] +  '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir+'log', 'a') as f:
        f.write('\n\n' + str(t_mark) + '\nsave dir:' + save_dir + '\n' + str(train_param) + '\n' + str(model_param) + '\n')
        
    print('start training')
    train(train_data, val_data, model, optimizer, batch_size, epochs, last_epoch, val_rate, teacher_forcing_ratio, save_dir, t, model_param, train_param)


if __name__ == '__main__':
    main()