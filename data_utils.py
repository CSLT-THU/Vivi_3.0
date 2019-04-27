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

        source_words = ('START1 ' + source + ' END1').split(' ')  # 用了START1和END1
        target = target[:-2] if target.find('\r\n') > -1 else target[:-1]
        target_words = target.replace('\t', ' / ').split(' ') + ['/'] + target.split('\t')[0].split(' ')  # 用5个句子训练

        source_ids = [word2id.get(word, vocab_size - 1) for word in source_words]  # default = 4776 '-' ?
        target_ids = [word2id.get(word, vocab_size - 1) for word in target_words]
        target_ids.insert(0, SOS_token)  # jiyuan没有
        target_ids.append(EOS_token)

        pairs_li.append([source_ids, target_ids])

    print('read traning set done')
    return pairs_li  # tmp


def read_test_data(file):
    print('read test set')
    input_li = []
    lines = []
    for line in open(file, 'r', encoding='utf-8').readlines():
        line = line.replace('\n', '') 
        lines.append(line.replace(' ', '')) # 打印用
        input_words = line.split(' ')
        input_ids = [word2id.get(word, vocab_size - 1) for word in input_words]  # default = 4776 '-' ?
        input_ids.insert(0, SOS_token)
        input_ids.append(EOS_token)
        input_li.append(input_ids)
    print('read test set done')
    return input_li, lines


def get_keywords(keywords):
    input_ids = []
    for word in keywords:
        input_ids.append(word2id.get(word, vocab_size - 1))
    input_ids.insert(0, SOS_token)
    input_ids.append(EOS_token)
    return [input_ids], [keywords]


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