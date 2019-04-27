# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import data_utils

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

# file paths
def get_options():
    options = {}
    options['yunmu_file'] = 'resource/YunLv/yun_utf8.txt'
    options['pingshui_file'] = ['resource/YunLv/ze_format.txt',
                                'resource/YunLv/ping_format.txt']  # old
    options['pingshui_file_new'] = ['resource/YunLv_new/words_z.txt',
                                    'resource/YunLv_new/words_p.txt']  # new
    # options['pingshui_file_new'] = ['resource/YunLv/yun_z_new.txt',
    #                                 'resource/YunLv/yun_p_new.txt']  # new
    options['hanzipinyin_file'] = 'resource/YunLv/hzpy.txt'
    return options
    
options = get_options()

class yunLv(object):
    """docstring for yunMu"""

    # sheng = ['b','p','m','f','d','t','n','l','g','k','h','j','q','x','z','c','s','r','zh','ch','sh']
    def __init__(self, options):

        file_path = options['yunmu_file']
        pingshui_yun_file_ping = options['pingshui_file'][1]
        pingshui_yun_file_ze = options['pingshui_file'][0]
        pingshui_yun_file_ping_new = options['pingshui_file_new'][1]
        pingshui_yun_file_ze_new = options['pingshui_file_new'][0]
        hanzipinyin_file_path = options['hanzipinyin_file']

        # 将如下押韵的字保存在self.hanzipinyin中。作为额外的韵的检查
        yun_constrain = [['zhi', 'chi', 'shi'], ['zi', 'ci', 'si']]
        self.hanzipinyin = [set([]) for i in range(len(yun_constrain))]
        with open(hanzipinyin_file_path, encoding='utf-8') as f:
            lines = f.readlines()
            for l in lines:
                l_temp = l.strip()
                l_split = l_temp.split(',')
                for i in range(len(yun_constrain)):
                    if l_split[1] in yun_constrain[i]:
                        self.hanzipinyin[i].add(l_split[0])

        # yundiao
        if 1:
            vocSet = set([])  # all words
            yunDict = {}  # all yun

            # get pingze, yun for each word (11729)
            f = open(file_path, encoding='utf-8').readlines()
            for l in f:
                ls = l.split()
                if len(ls) == 3:
                    word = ls[0]
                    pingZ = ls[1]
                    yun = ls[2]

                    vocSet.add(word)
                    yunDict[word] = {'p': pingZ, 'y': yun}
                else:
                    pass

            self.vocSet = vocSet
            self.yunDict = yunDict

        if 1:
            # vocab set in pingshuiyun
            self.vocSet_ping = set([])
            for f in options['pingshui_file']:
                for l in open(f, encoding='utf-8').readlines():
                    for w in l:
                        self.vocSet_ping.add(w)

            # yunlist, used in function getYunLineLen
            self.pzlist = {}
            self.pzlist['p'] = set([])
            self.pzlist['z'] = set([])
            self.pzlist['0'] = set([])

            self.yunlist = []

            for l in open(pingshui_yun_file_ze, encoding='utf-8').readlines():
                temp_list = set([])
                for w in l:
                    self.pzlist['z'].add(w)
                    temp_list.add(w)
                self.yunlist.append(temp_list)

            for l in open(pingshui_yun_file_ping, encoding='utf-8').readlines():
                temp_list = set([])
                for w in l:
                    self.pzlist['p'].add(w)
                    temp_list.add(w)
                self.yunlist.append(temp_list)

            # yunlist_new, used in function yapingshui
            self.yunlist_new = []
        self.yun_pzlist = {}
        self.yun_pzlist['p'] = set([])
        self.yun_pzlist['z'] = set([])

        for l in open(pingshui_yun_file_ze_new, encoding='utf-8').readlines():
            temp_list = set([])
            for w in l.strip():
                self.yun_pzlist['z'].add(w)
                temp_list.add(w)
            self.yunlist_new.append(temp_list)

            for l in open(pingshui_yun_file_ping_new, encoding='utf-8').readlines():
                temp_list = set([])
                for w in l.strip():
                    self.yun_pzlist['p'].add(w)
                    temp_list.add(w)
                self.yunlist_new.append(temp_list)

    def getYunDiao(self, x):
        y = 'aaaa'
        p = '-1'
        if x in self.vocSet:
            p = self.yunDict[x]['p']
            y = self.yunDict[x]['y']
        return {'y': y, 'p': p}

    def yapingshui(self, word1, word):
        sig_bool = False
        for temp_set in self.yunlist_new:
            if (word1 in temp_set) and (word in temp_set):
                sig_bool = True
                # 如果不符合yun_contrain的韵则不行。（一个在一个不在）
                for yun_cons_set in self.hanzipinyin:
                    if (word1 in yun_cons_set) and (word not in yun_cons_set):
                        sig_bool = False
                    if (word1 not in yun_cons_set) and (word in yun_cons_set):
                        sig_bool = False
        return sig_bool

    def getYunLineLen(self, word):
        for temp_set in self.yunlist:
            if word in temp_set:
                return len(temp_set)
        return 0

yunmuModel = yunLv(options)

def get_next_word(decoder_output, decoded_words):
    # topv, topi = decoder_output.data.topk(1)
    sorted, indices = torch.sort(decoder_output, descending=True)
    id = torch.Tensor([1])  # default=END
    word = 'N'
    indices = torch.flatten(indices)
    # indices = indices.tolist()
    sen_num = len(decoded_words) // sen_len + 1 # 当前是第几句话
    w_num = len(decoded_words) % sen_len + 1 #当前是一句话的第几个字
    candidates = [] # for developing use
    for i in range(2000): # max_find
        candidate_word = id2word[str(indices[i].item())]
        candidates.append(candidate_word) # 
        
        # forbidden words
        if candidate_word in forbidden_words:
            continue        
        # low frequency
        if word2count[candidate_word] < 200 or (candidate_word not in word2count.keys()):
            continue     
        # no repeat
        if candidate_word in decoded_words:
            continue
        
        # Yun
        if sen_num in yun_sen[1:] and w_num == sen_len:
            yun_word = decoded_words[yun_sen[0]*sen_len-1] # 第一个押韵句子的最后一个字
            if not yunmuModel.yapingshui(candidate_word, yun_word):
                continue
        # Lv
        candidate_lv = yunmuModel.getYunDiao(candidate_word)['p']
        target_lv = lv_list[sen_num-1][w_num-1] 
        if target_lv != '0' and candidate_lv != target_lv:
            continue
        
        id = indices[i]
        word = candidate_word 
        # print(i+1)
        # print(word)
        break
        
    if word == 'N': # if cannot meet all requirements
        print(candidates)
    return id, word
