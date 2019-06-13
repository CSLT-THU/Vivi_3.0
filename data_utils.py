# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import torch
import json
import pickle
from sklearn import model_selection

from word_emb import emb_size, word2id, id2word, emb, word2count, vocab_size, SOS_token, EOS_token, PAD_token, UNK_token
# from planning.plan import planner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = True if torch.cuda.is_available() else False


def read_train_data(file):
    print('read training set')
    # pairs_tensor = []
    pairs_li = []
    lines = 0  # 一共多少个训练数据

    for line in open(file, 'r', encoding='utf-8').readlines():
        lines += 1
        try:
            source, target = line.split('==')
        except:
            print('format mismatch in dataset: ', line.split('=='))
            continue

        if len(source.split(' - ')) == 4: # poem_1031k_theme 
            source = source + ' - ' + source.split(' - ')[0]  
        source_words = ('START1 ' + source + ' END1').split(' ') 
        target = target.replace('\n', '')
        target_words = target.replace('\t', ' / ').split(' ') + ['/'] + target.split('\t')[0].split(' ')  # 用5个句子训练

        source_ids = [word2id.get(word, vocab_size - 1) for word in source_words]  # default = 4776 '-' PAD?
        target_ids = [word2id.get(word, vocab_size - 1) for word in target_words]
        target_ids.append(EOS_token) # 没有SOS_token

        if len(target_ids) == 40: # 只考虑七言
            pairs_li.append([source_ids, target_ids])

    print('training set size:', len(pairs_li))
    print('read traning set done')
    return pairs_li  # tmp

def read_train_data_2(file):
    print('read training set')
    # pairs_tensor = []
    pairs_li = []
    lines = 0  # 一共多少个训练数据

    for line in open(file, 'r', encoding='utf-8').readlines():
        lines += 1
        try:
            source, target = line.split('==')
        except:
            print('format mismatch in dataset: ', line.split('=='))
            continue

        source_1 = source.split(' - ')[0] + ' - ' +  source.split(' - ')[1]
        source_2 = source.split(' - ')[2] + ' - ' + source.split(' - ')[3]
        source_words_1 = ('START1 ' + source_1 + ' END1').split(' ')
        source_words_2 = ('START1 ' + source_2 + ' END1').split(' ')
        source_ids_1 = [word2id.get(word, vocab_size - 1) for word in source_words_1]
        source_ids_2 = [word2id.get(word, vocab_size - 1) for word in source_words_2]

        target = target.replace('\n', '')
        target_1 = target.split('\t')[0] + ' / ' + target.split('\t')[1]
        target_2 = target.split('\t')[2] + ' / ' + target.split('\t')[3]
        target_words_1 = (target_1 + ' END').split(' ')  # 用5个句子训练
        target_words_2 = (target_2 + ' END').split(' ')
        target_ids_1 = [word2id.get(word, vocab_size - 1) for word in target_words_1]
        target_ids_2 = [word2id.get(word, vocab_size - 1) for word in target_words_2]

        if len(target_ids_1) == 16: # 只考虑七言
            pairs_li.append([source_ids_1, target_ids_1])
        if len(target_ids_2) == 16:
            pairs_li.append([source_ids_2, target_ids_2])

    print('training set size:', len(pairs_li))
    return pairs_li  # tmp


def line2ids(line): # for testset
    input_words = line.split(' ')
    input_ids = [word2id.get(word, vocab_size - 1) for word in input_words]  # default = 4776 '-' ?
    input_ids.insert(0, SOS_token)
    input_ids.append(EOS_token)
    return input_ids


def plan(line):
    text = line.replace(' ', '')
    # text = line.replace(' ', '').replace('-', '') # 关键字全部连在一起
    keywords = planner.plan(text)
    new_line = ''
    for keyword in keywords:
        for word in keyword:
            new_line += word + ' '
        new_line += '- '
    new_line = new_line.rstrip(' - ')
    return new_line


def get_line(line, lines, use_planning):
    if not use_planning:
        lines.append(line.replace(' ', ''))  # 打印用
    else:
        line_tmp = line
        line = plan(line)
        lines.append(line_tmp.replace(' ', '') + ' == ' + line.replace(' ', ''))
    return line, lines


def read_test_data(file, use_planning):
    print('read test set')
    input_li = []
    lines = []
    for line in open(file, 'r', encoding='utf-8').readlines():
        line = line.replace('\n', '') 
        line, lines = get_line(line, lines, use_planning)
        input_ids = line2ids(line)
        input_li.append(input_ids)
    print('read test set done')
    return input_li, lines

def read_eval_data(file, use_planning):
    print('read eval set')
    input_li = []
    lines = []
    targets = []
    for line in open(file, 'r', encoding='utf-8').readlines():
        line = line.replace('\n', '') 
        line, target = line.split('==')
        
        target = target.replace(' ', '').replace('\t', '/')
        targets.append(target)

        line, lines = get_line(line, lines, use_planning)
        input_ids = line2ids(line)
        input_li.append(input_ids)
    print('read test set done')
    return input_li, lines, targets


def read_eval_data_2(file, use_planning):
    print('read eval set')
    input_li_1 = [] #
    input_li_2 = [] #
    lines = []
    targets = []
    for line in open(file, 'r', encoding='utf-8').readlines():
        line = line.replace('\n', '')
        line, target = line.split('==')

        target = target.replace(' ', '').replace('\t', '/')
        targets.append(target)

        line, lines = get_line(line, lines, use_planning)
        #
        line_li = line.split(' - ')
        line1 = line_li[0] + ' - ' + line_li[1]
        line2 = line_li[2] + ' - ' + line_li[3]
        # line1 = line_li[0] + ' - ' + line_li[1] + ' - ' + line_li[0] + ' - ' + line_li[1]
        # line2 = line_li[2] + ' - ' + line_li[3] + ' - ' + line_li[2] + ' - ' + line_li[3]
        input_ids_1 = line2ids(line1)
        input_li_1.append(input_ids_1)
        input_ids_2 = line2ids(line2)
        input_li_2.append(input_ids_2)
    print('read test set done')
    return (input_li_1, input_li_2), lines, targets


def get_keywords(keywords, use_planning):
    input_li = []
    lines = []
    line = keywords
    line, lines = get_line(line, lines, use_planning)
    input_ids = line2ids(line)
    input_li.append(input_ids)
    return input_li, lines


def split_dataset(pairs, val_rate):
    train_set = pairs
    val_set = None
    if val_rate:
        train_set, val_set = model_selection.train_test_split(train_set, test_size=val_rate, random_state=None)
    return train_set, val_set


def sort_batch_data(batch_x, x_len, batch_y, y_len):  # 按x的长度排序
    sorted_x_len, sorted_id = x_len.sort(dim=0, descending=True)
    sorted_x = batch_x[sorted_id]
    sorted_y = batch_y[sorted_id]
    sorted_y_len = y_len[sorted_id]
    return sorted_x, sorted_x_len, sorted_y, sorted_y_len