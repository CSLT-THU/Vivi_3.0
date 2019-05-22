# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import torch
import json

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
        if candidate_word not in word2count.keys():
            continue     
        elif word2count[candidate_word] < 200:
            continue
            
        # no repeat
        if candidate_word in decoded_words:
            continue
        
        # Yun
        if sen_num in yun_sen[1:] and w_num == sen_len:
            yun_word = decoded_words[yun_sen[0]*sen_len-1] # 第一个押韵句子的最后一个字
            if not word_dict[candidate_word]['yun'] == word_dict[yun_word]['yun']:
                continue
        # Lv
        candidate_lv = word_dict[candidate_word]['pz']
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
