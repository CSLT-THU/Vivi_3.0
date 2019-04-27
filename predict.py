# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import time
import os

import importlib
import configparser
import torch.utils.data as Data
from data_utils import read_test_data, get_keywords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

######################################################################
# predict
# ==========

def predict(test_data, lines, model, model_param, ckpt_path, poem_type, cangtou, assign_yun):
    model.eval()
    context = ''
    for i, data in enumerate(test_data):
        # 每一步 loader 释放一小批数据用来学习，step=总数据量/batch_size，enumerate把每次提取编写索引。
        # batch_x: B*T tensor

        output_words = model.predict(data, cangtou)

        output_words.insert(7, '/')
        output_words.insert(15, '/')
        output_words.insert(23, '/')
        output_sentence = ' '.join(output_words)
        print((i+1), lines[i], ' ==== ', output_sentence)
        context = context + lines[i] + ' ==== ' + output_sentence + '\n'
        
    # logs 
    file = 'result/result_' + ckpt_path.split('/')[1].split('.pkl')[0] + '.txt'
    with open(file, 'a', encoding='utf-8') as f:
        t = time.strftime("%m-%d", time.localtime())
        f.write(t)
        f.write('\n'+context+'\n')


def main():
    # ========= Get Parameter =========#
    conf = configparser.ConfigParser()
    conf.read('config.ini', encoding="utf-8-sig")

    model_name = conf.get('train', 'model')
    ckpt_path = conf.get('predict', 'ckpt_path')
    test_set = conf.get('predict', 'test_set')
    use_planning = conf.get('predict', 'use_planning')
    poem_type = conf.get('predict', 'poem_type')
    cangtou = conf.get('predict', 'cangtou')
    keywords = conf.get('predict', 'keywords')
    assign_yun = conf.get('predict', 'assign_yun')
    
    model_param_li = conf.items(model_name)
    model_param = {}
    for item in model_param_li:
        model_param[item[0]] = item[1]


    # ========= Preparing Data =========#

    # read data
    if cangtou:
        test_set, lines = get_keywords(cangtou)
    elif keywords:
        test_set, lines = get_keywords(keywords)
    else:    
        test_set, lines = read_test_data(test_set)
    
    # 实例化
    data_path = 'models.' + model_name + '.PoetryData'
    PoetryData = importlib.import_module(data_path)
    test_Dataset = getattr(PoetryData, 'PoetryData')(test_set, src_max_len=int(model_param['input_max_len']), 
                                                     tgt_max_len=int(model_param['target_max_len']), test=True)
    
    # 变成小批
    test_data = Data.DataLoader(
        dataset=test_Dataset,  # torch TensorDataset format
        batch_size=1,  
        shuffle=False,  
        # num_workers=2,  # 多线程来读数据，提取xy的时候几个数据一起提取
    )

    # ========= Preparing Model =========#

    model_path = 'models.' + model_name + '.' + model_name
    Model = importlib.import_module(model_path)  # 导入模块
    model = getattr(Model, model_name)(model_param)  # 反射并实例化
    print('model:', model)

    # load model
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'])
        predict(test_data, lines, model, model_param, ckpt_path, poem_type, cangtou, assign_yun)
    else: 
        print('ckpt_path does not exist.')


if __name__ == '__main__':
    main()
    