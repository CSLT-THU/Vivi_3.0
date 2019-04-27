# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import json

class WordEmb():
    '''
    id2word = {'0':'START' '1':'END' '2':'/' '3':'START1' '4':'END1' '4776':'-' '4777':'UNK'} len=4778
    word2id len=4777(没有'END1') 
    emb len=4777
    word2count: 4764 来自数据集统计
    '''
    def __init__(self):
        with open('resource/word_emb.json', 'r', encoding='utf-8') as f1:
            dict = json.load(f1)
            self.emb_size = 200
            self.word2id = dict['word2id']
            self.id2word = dict['id2word']
            self.emb = dict['emb']
            self.word2count = dict['word2count']
            self.vocab_size = dict['vocab_size']
            self.SOS_token = dict['SOS_token']
            self.EOS_token = dict['EOS_token']
            self.PAD_token = dict['PAD_token']
            self.UNK_token = dict['UNK_token']

    def get(self):
        return self.emb_size, self.word2id, self.id2word, self.emb, self.word2count, self.vocab_size, \
               self.SOS_token, self.EOS_token, self.PAD_token, self.UNK_token

wordemb = WordEmb()
emb_size, word2id, id2word, emb, word2count, vocab_size, SOS_token, EOS_token, PAD_token, UNK_token = wordemb.get()
