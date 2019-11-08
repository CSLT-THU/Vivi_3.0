# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import torch
import json

from word_emb import emb_size, word2id, id2word, emb, word2count, vocab_size, SOS_token, EOS_token, PAD_token, UNK_token

poem_type = 'poem7' # 需要修改
hard_lv = False # 强lv

forbidden_words = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '千', '百', '万',
                   '艇', '些', '的',
                   'START', 'END', '/', 'START1', 'END1', '-', 'UNK'] # END1
# forbidden_id = [word2id[word] for word in forbidden_words]
# yun_sen = [2,4]

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
        lv_list = [['0', '0', '0', '-1', '0', '1', '0'], ['0', '-1', '0', '1', '0', '-1', '0'],
                   ['0', '-1', '0', '1', '0', '-1', '0'], ['0', '1', '0', '-1', '0', '1', '0']]
        # lv_list = [['0', 'p', '0', 'z', '0', 'p', '0'], ['0', 'z', '0', 'p', '0', 'z', 'p'],
        #             ['0', 'z', '0', 'p', '0', 'z', '0'], ['0', 'p', '0', 'z', '0', 'p', 'p']]
yun_li = [
    ['a', 'ia', 'ua'],
    ['ai', 'uai'],
    ['an', 'ian', 'uan'],
    ['ang', 'iang', 'uang'],
    ['ao', 'iao'],
    ['e', 'o', 'uo'], # not (y)e
    ['ei', 'ui'],
    ['en', 'in', 'un'],
    ['eng', 'ing', 'ong', 'iong'],
    ['i', 'er'], 
    ['i1'], # z c s 
    ['i2'], # zh ch sh
    ['ie', 'e1'], # (y)e
    ['ou', 'iu'],
    ['u','v'], # 本应分开
    ['ve', 'ue'] 
]

with open('resource/word_dict.json', 'r', encoding='utf-8') as f1:
    word_dict = json.load(f1)


def is_rhyme(word1, word2):
    yun1 = -1
    yun2 = -2
    for i in range(len(yun_li)):
        if word_dict[word1]['yun'] in yun_li[i]:
            yun1 = i
        if word_dict[word2]['yun'] in yun_li[i]:
            yun2 = i
    if yun1 == yun2:
        return True
    else:
        return False
    
    
def get_next_word(decoder_output, decoded_words, hard_rhyme, hard_tone):
    # topv, topi = decoder_output.data.topk(1)
    sorted, indices = torch.sort(decoder_output, descending=True)
    id = torch.Tensor([1])  # default=END
    word = 'N'
    indices = torch.flatten(indices)
    # indices = indices.tolist()
    sen_num = len(decoded_words) // sen_len + 1 # 当前是第几句话 1-4
    w_num = len(decoded_words) % sen_len + 1 #当前是一句话的第几个字 1-7
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
        
        if hard_rhyme: # Yun  
            yun_word = None
            if w_num == sen_len and sen_num == 2:  # 第二句最后一个字
                word0 = decoded_words[sen_len - 1]
                if word_dict[word0]['pz'] == 'p':
                    yun_word = word0
                    if not is_rhyme(candidate_word, yun_word):
                        continue
                
            if w_num == sen_len and sen_num == 4:  # 第四句最后一个字
                if yun_word == None:
                    yun_word = decoded_words[sen_len*2 - 1]
                if not is_rhyme(candidate_word, yun_word):
                    continue
                
                
            # if sen_num in yun_sen[1:] and w_num == sen_len:
            #     yun_word = decoded_words[yun_sen[0] * sen_len - 1]  # 第一个押韵句子的最后一个字
            #     if not is_rhyme(candidate_word, yun_word):
            #         continue
        
        if w_num + (sen_num - 1) * sen_len > 2 and hard_tone:
            candidate_lv = word_dict[candidate_word]['pz']
            word2 = decoded_words[1]
            word2_lv = word_dict[word2]['pz']
            if lv_list[sen_num - 1][w_num - 1] == '1':
                target_lv = word2_lv
                if candidate_lv != target_lv:
                    continue
            elif lv_list[sen_num - 1][w_num - 1] == '-1':
                if word2_lv == 'p':
                    target_lv = 'z'
                elif word2_lv == 'z':
                    target_lv = 'p'
                else:
                    target_lv = '0'
                if candidate_lv != target_lv:
                    continue
                
         
        # if hard_tone: # Lv
        #     candidate_lv = word_dict[candidate_word]['pz']
        #     target_lv = lv_list[sen_num - 1][w_num - 1]
        #     if target_lv != '0' and candidate_lv != target_lv:
        #         continue
        
        id = indices[i]
        word = candidate_word 
        break
        
    if word == 'N': # if cannot meet all requirements
        id = indices[0]
        word = id2word[str(indices[0].item())]
        # print(candidates)
    return id, word

def get_yun(word):
    return word_dict[word]['yun']

def get_lv(word):
    return word_dict[word]['pz']