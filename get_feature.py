# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import torch
import json
import numpy as np

from word_emb import emb_size, word2id, id2word, emb, word2count, vocab_size, SOS_token, EOS_token, PAD_token, UNK_token

poem_type = 'poem7' # 需要修改
hard_lv = True # 强lv

forbidden_words = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '千', '百', '万',
                   '艇', '些', '的',
                   'START', 'END', '/', 'START1', 'END1', '-', 'UNK'] # END1
# forbidden_id = [word2id[word] for word in forbidden_words]
yun_sen = [2,4]

lv_list = []
sen_len = 0

if poem_type == 'poem5':
    sen_len = 5
    lv_list = [['z', 'z', 'p', 'p', 'z'], ['p', 'p', 'z', 'z', 'p'], 
                ['p', 'p', 'p', 'z', 'z'], ['z', 'z', 'z', 'p', 'p']]
    
elif poem_type == 'poem7':
    sen_len = 7
    if hard_lv:
        lv_list = [['p', 'p', 'z', 'z', 'p', 'p', 'z'], ['z', 'z', 'p', 'p', 'z', 'z', 'p'],
                    ['z', 'z', 'p', 'p', 'p', 'z', 'z'], ['p', 'p', 'z', 'z', 'z', 'p', 'p']]
    else:
        lv_list = [['0', 'p', '0', 'z', '0', 'p', '0'], ['0', 'z', '0', 'p', '0', 'z', 'p'],
                    ['0', 'z', '0', 'p', '0', 'z', '0'], ['0', 'p', '0', 'z', '0', 'p', 'p']]

with open('resource/word_dict.json', 'r', encoding='utf-8') as f1:
    word_dict = json.load(f1)

all_lv = []
for i in range(vocab_size):
    word = id2word[str(i)]
    if word in word_dict.keys():
        word_lv = word_dict[word]['pz']
    else:
        word_lv = ''
    all_lv.append(word_lv)
all_lv = np.array(all_lv)

all_yun = []
for i in range(vocab_size):
    word = id2word[str(i)]
    if word in word_dict.keys():
        word_yun = word_dict[word]['yun']
    else:
        word_yun = ''
    all_yun.append(word_yun)
all_yun = np.array(all_yun)
 
def get_feature(decoded_words, sen_num, w_num, batch_size):
    feature1 = []
    feature2 = []
    for i in range(batch_size):
        target_lv = lv_list[sen_num][w_num]
        target_lv = [target_lv] * vocab_size
        target_lv = np.array(target_lv)
        feature1_batch = np.where(all_lv == target_lv, 1.0, 0.0)
        feature1.append(feature1_batch)

        decoded_words_batch = decoded_words[i]
        if sen_num+1 in yun_sen[1:] and w_num+1 == sen_len:
            yun_word = decoded_words_batch[yun_sen[0] * (sen_len+1) - 2]  # 第一个押韵句子的最后一个字
            if yun_word in word_dict.keys(): # 生成标志词需要特殊处理，待考虑
                target_yun = word_dict[yun_word]['yun']
                target_yun = [target_yun] * vocab_size
                target_yun = np.array(target_yun)
                feature2_batch = np.where(all_yun == target_yun, 1.0, 0.0)
            else:
                feature2_batch = np.zeros(vocab_size)
        else:
            feature2_batch = np.zeros(vocab_size)
        feature2.append(feature2_batch)
    
    feature1 = np.array(feature1).astype(np.float32)
    feature2 = np.array(feature2).astype(np.float32)
    
    return feature1, feature2