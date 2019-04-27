# -*- coding: utf-8 -*-
import json

with open("../embedding/word2id.json",'r') as f0:
    dic = json.load(f0)
    emb_vocab = list(dic.keys())
emb_len = len(emb_vocab)

hzpy_vocab = []
with open('hzpy.txt', 'r', encoding='utf-8') as f1:
    lines = f1.readlines()
    for line in lines:
        word = line[0]
        hzpy_vocab.append(word)
hzpy_len = len(hzpy_vocab)

yun_utf8_vocab = []
with open('yun_utf8.txt', 'r', encoding='utf-8') as f5:
    lines = f5.readlines()
    for line in lines:
        word = line[0]
        yun_utf8_vocab.append(word)
yun_utf8_len = len(yun_utf8_vocab)

p_old_vocab = []
with open('ping_format.txt', 'r', encoding='utf-8') as f2:
    lines = f2.readlines()
    for line in lines:
        for word in line:
            p_old_vocab.append(word)
p_old_len = len(p_old_vocab)

z_old_vocab = []
with open('ze_format.txt', 'r', encoding='utf-8') as f3:
    lines = f3.readlines()
    for line in lines:
        for word in line:
            z_old_vocab.append(word)
z_old_len = len(z_old_vocab)

p_new_vocab = []
with open('yun_p_new.txt', 'r', encoding='utf-8') as f4:
    lines = f4.readlines()
    for line in lines:
        for word in line:
            p_new_vocab.append(word)
p_new_len = len(p_new_vocab)

z_new_vocab = []
with open('yun_z_new.txt', 'r', encoding='utf-8') as f4:
    lines = f4.readlines()
    for line in lines:
        for word in line:
            z_new_vocab.append(word)
z_new_len = len(z_new_vocab)

print('''
emb_len:{} 
hzpy_len:{} 
yun_utf8_len:{}
p_old_len:{}
z_old_len:{}
p_new_len:{}
z_new_len:{}
'''.format(emb_len, hzpy_len, yun_utf8_len, p_old_len, z_old_len, p_new_len, z_new_len))

emb_vocab = set(emb_vocab)
hzpy_vocab = set(hzpy_vocab)
yun_utf8_vocab = set(yun_utf8_vocab)
p_old_vocab = set(p_old_vocab)
z_old_vocab = set(z_old_vocab)
p_new_vocab = set(p_new_vocab)
z_new_vocab = set(z_new_vocab)

oov = emb_vocab - hzpy_vocab
print(oov)
print(len(oov))

oov5 = emb_vocab - yun_utf8_vocab
print(oov5)
print(len(oov5))

print(oov5 & oov)

oov1 = (emb_vocab - p_old_vocab) & (emb_vocab - z_old_vocab) 
print(oov1)
print(len(oov1))

oov2 = (emb_vocab - p_new_vocab) & (emb_vocab - z_new_vocab) 
print(oov2)
print(len(oov2))

oov3 = oov & oov2
print(oov3)
print(len(oov3))

print(p_new_vocab & z_new_vocab)