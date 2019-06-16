# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import time
import os
import json
import argparse

import importlib
import configparser
import torch.utils.data as Data
from data_utils import read_test_data, get_keywords, read_eval_data, read_eval_data_2 ###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

# word dict
with open('resource/word_dict.json', 'r', encoding='utf-8') as f1:
    word_dict = json.load(f1)
    
######################################################################
# predict
# ==========

def predict(test_data, lines, targets, model, predict_param, ckpt_path, poem_type, cangtou):
    model.eval()
    context = ''
    for i, data in enumerate(test_data):
        # 每一步 loader 释放一小批数据用来学习，step=总数据量/batch_size，enumerate把每次提取编写索引。
        # batch_x: B*T tensor

        output_words = model.predict(data, cangtou, predict_param)
        
        output_words.insert(7, '/') # 改
        output_words.insert(15, '/')
        output_words.insert(23, '/')
        
        output_sentence = ''.join(output_words)
        context_poem = str(i+1) + '\n' + lines[i] + ' ==== ' + output_sentence + '\n'
        ###
        word1 = output_words[14]
        word2 = output_words[30]
        yun1 = word_dict[word1]['yun']
        yun2 = word_dict[word2]['yun']
        
        context = context + context_poem
        print(context_poem, yun1, yun2)
        if targets:
            target = lines[i] + ' ==== ' + targets[i] + '\n'
            context = context + target
            print(target)
        
    # logs 
    file = 'result/result_' + ckpt_path.split('/')[1].split('.pkl')[0] + '.txt'
    with open(file, 'a', encoding='utf-8') as f:
        t = time.strftime("%m-%d", time.localtime())
        f.write(t)
        f.write('\n'+context+'\n')


def main():
    # ========= Loading Params =========#
    # predict param
    parser = argparse.ArgumentParser(description='Vivi')

    parser.add_argument('--model_name', type=str, default='Transformer')
    parser.add_argument('--ckpt_path', type=str, default='ckpt/06-16_Transformer_epoch=1_loss=340.1.pkl')
    # parser.add_argument('--model_name', type=str, default='Seq2seq_7')
    # parser.add_argument('--ckpt_path', type=str, default='ckpt/06-10_Seq2seq_7_epoch=2_loss=157.7.pkl')
    parser.add_argument('--cangtou', type=str, default='')
    parser.add_argument('--keywords', type=str, default='')
    parser.add_argument('--test_set', type=str, default='resource/dataset/testset.txt')
    parser.add_argument('--eval_set', type=str, default='')
    parser.add_argument('--use_planning', type=bool, default=False)
    parser.add_argument('--bleu_eval', type=bool, default=False)
    parser.add_argument('--poem_type', type=str, default='poem7')

    args = parser.parse_args()

    model_name = args.model_name
    ckpt_path = args.ckpt_path
    cangtou = args.cangtou
    keywords = args.keywords
    test_set = args.test_set
    eval_set = args.eval_set
    use_planning = args.use_planning
    bleu_eval = args.bleu_eval
    poem_type = args.poem_type
    
    predict_param = vars(args)
    print('predict param: ', predict_param)
    
    # model param
    conf = configparser.ConfigParser()
    conf.read('config/config_'+model_name+'.ini')
    model_param_li = conf.items('model_param')
    model_param = {}
    for item in model_param_li:
        model_param[item[0]] = item[1]

    # ========= Preparing Data =========#

    # read data
    targets = None
    if cangtou:
        test_set, lines = get_keywords(cangtou, use_planning)
    elif keywords:
        test_set, lines = get_keywords(keywords, use_planning)
    elif test_set:    
        test_set, lines = read_test_data(test_set, use_planning)
    else: # eval
        test_set, lines, targets = read_eval_data(eval_set, use_planning) # read_eval_data May24
        
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
        collate_fn=PoetryData.collate_fn # Jun16
        # num_workers=2,  # 多线程来读数据，提取xy的时候几个数据一起提取
    )

    # ========= Preparing Model =========#
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        # model_param = checkpoint['settings']     # Jun16
        model_param = checkpoint['model_param']
        model_path = 'models.' + model_name + '.' + model_name
        Model = importlib.import_module(model_path)  # 导入模块
        model = getattr(Model, model_name)(model_param)  # 反射并实例化
        print('model:', model)
        
        model.load_state_dict(checkpoint['model'])
        
        if 'train_param' in checkpoint.keys():
            print('train param:', checkpoint['train_param'])
        else:
            print('train param not recorded.')
        
        predict(test_data, lines, targets, model, predict_param, ckpt_path, poem_type, cangtou)
    else: 
        print('ckpt_path does not exist.')


if __name__ == '__main__':
    main()
    