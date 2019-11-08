# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import time
import os
import json
import argparse
import random

import importlib
import configparser
import torch.utils.data as Data
import data_utils
import constrains

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

# seed = 1
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True

# word dict
with open('resource/word_dict.json', 'r', encoding='utf-8') as f1:
    word_dict = json.load(f1)

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
        context_poem = lines[i] + ' ==== ' + output_sentence + '\n'
        # context_poem = str(i + 1) + '\n' + lines[i] + ' ==== ' + output_sentence + '\n' # Jul3

        context = context + context_poem
        if i < 3:
            print(context_poem)
        if targets:
            target = lines[i] + ' ==== ' + targets[i] + '\n'
            # context = context + target # Jul3
            if i < 3:
                print(target)
    
    # logs 
    if len(ckpt_path.split('/'))==2:
        file = 'result/' + ckpt_path.split('/')[1].split('.pkl')[0] + '_' + str(time.strftime("%H%M%S", time.localtime())) + '.txt'
    else:
        file = 'result/' + ckpt_path.split('/')[2].split('.pkl')[0] + '_' + str(time.strftime("%H%M%S", time.localtime())) + '.txt'
    with open(file, 'w', encoding='utf-8') as f:
        f.write(context)
    print('result saved at:', file)
    
    return file
    

def pred(predict_param):
    # ========= Loading Params =========#
    # predict param
    parser = argparse.ArgumentParser(description='Vivi')

    parser.add_argument('--model_name', type=str, default='Seq2seq_12')
    # parser.add_argument('--ckpt_path', type=str, default='ckpt/0802164334_Seq2seq_12_ep=3_loss=150.38.pkl')
    # parser.add_argument('--ckpt_path', type=str, default='ckpt/0802164334_Seq2seq_12_ep=4_loss=149.02.pkl')
    parser.add_argument('--ckpt_path', type=str, default='ckpt/0802164555_Seq2seq_12_ep=3_loss=154.49.pkl')
    # parser.add_argument('--ckpt_path', type=str, default='ckpt/0802164555_Seq2seq_12_ep=4_loss=152.79.pkl')
    # parser.add_argument('--ckpt_path', type=str, default='ckpt/0802164555_Seq2seq_12_ep=5_loss=151.01.pkl')
    # parser.add_argument('--ckpt_path', type=str, default='ckpt/0802164555_Seq2seq_12/0802164555_Seq2seq_12_ep=6_loss=150.18.pkl')
    parser.add_argument('--cangtou', type=str, default='')
    parser.add_argument('--keywords', type=str, default='')
    parser.add_argument('--test_set', type=str, default='')
    parser.add_argument('--eval_set', type=str, default='resource/dataset/poem_1031k_theme_test_1k.txt')
    parser.add_argument('--use_planning', type=bool, default=False)
    parser.add_argument('--bleu_eval', type=bool, default=False)
    parser.add_argument('--poem_type', type=str, default='poem7')
    parser.add_argument('--train_mode', type=str, default='kw2poem') # nL21L or kw2poem
    parser.add_argument('--note', type=str, default='')

    parser.add_argument('--as_train', type=bool, default=False) # Jul23
    parser.add_argument('--pred_soft', type=bool, default=True) 
    parser.add_argument('--template', type=bool, default=False)  # 2 template T, 4 soft F  
    parser.add_argument('--hard_rhyme', type=bool, default=True)
    parser.add_argument('--hard_tone', type=bool, default=True)
    parser.add_argument('--w1', type=float, default=4.)
    parser.add_argument('--w2', type=float, default=0.)
   
    args = parser.parse_args()
    
    if predict_param == {}:
        print('pred param from args')
        predict_param = vars(args)
    else:
        print('pred param from py script')

    model_name = predict_param['model_name']
    ckpt_path = predict_param['ckpt_path']
    cangtou = predict_param['cangtou']
    keywords = predict_param['keywords']
    test_set = predict_param['test_set']
    eval_set = predict_param['eval_set']
    use_planning = predict_param['use_planning']
    poem_type = predict_param['poem_type']
    train_mode = predict_param['train_mode']
    as_train = predict_param['as_train']
    
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
        test_set, lines = data_utils.get_keywords(cangtou, use_planning)
    elif keywords:
        test_set, lines = data_utils.get_keywords(keywords, use_planning)
    elif test_set:    
        test_set, lines = data_utils.read_test_data(test_set, use_planning)
    elif train_mode == 'nL21L':
        test_set, lines, targets = data_utils.read_nL21L_eval_data(eval_set)
    else: # eval
        test_set, lines, targets = data_utils.read_eval_data(eval_set, use_planning) # read_eval_data May24
        
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
            # Jul23
            if as_train:
                train_param = checkpoint['train_param']
                predict_param['ckpt_path'] = ckpt_path
                predict_param['pred_soft'] = train_param['train_soft']
                predict_param['template'] = train_param['template']
                predict_param['w1'] = train_param['w1']
                predict_param['w2'] = train_param['w2']
            print('predict param: ', predict_param)
        else:
            print('train param not recorded.')

        save_file = predict(test_data, lines, targets, model, predict_param, ckpt_path, poem_type, cangtou)
    else: 
        print('ckpt_path does not exist.')
        save_file = None
        
    return save_file

def main():
    save_file = pred({})

if __name__ == '__main__':
    main()
    