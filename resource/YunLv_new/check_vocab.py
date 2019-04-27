# -*- coding: utf-8 -*-
import json

def get_emb():
    with open("../embedding/word2id.json", 'r') as f0:
        dic = json.load(f0)
        emb_vocab = list(dic.keys())
    emb_len = len(emb_vocab)
    return emb_vocab, emb_len

def get_hzpy():
    hzpy_vocab = []
    with open('hzpy.txt', 'r', encoding='utf-8') as f1:
        lines = f1.readlines()
        for line in lines:
            word = line[0]
            hzpy_vocab.append(word)
    hzpy_len = len(hzpy_vocab)
    return hzpy_vocab, hzpy_len

def get_yun_utf8():
    yun_utf8_vocab = []
    yun_utf8_vocab_li = []
    with open('yun_utf8.txt', 'r', encoding='utf-8') as f5:
        lines = f5.readlines()
        for line in lines:
            if len(line.split(' ')) == 3:
                word = line.split(' ')[0]
                pz = line.split(' ')[1]
                yun = line.split(' ')[2]
                yun_utf8_vocab_li.append(word)
                yun_utf8_vocab.append({'word':word, 'pz':pz, 'yun':yun})
            else:
                print('exception occurs')
                print(line.split(' '))
                continue
    yun_utf8_len = len(yun_utf8_vocab)
    return yun_utf8_vocab, yun_utf8_vocab_li, yun_utf8_len

def get_p_old():
    p_old_vocab = []
    with open('ping_format.txt', 'r', encoding='utf-8') as f2:
        lines = f2.readlines()
        for line in lines:
            for word in line:
                p_old_vocab.append(word)
    p_old_len = len(p_old_vocab)
    return p_old_vocab, p_old_len

def get_z_old():
    z_old_vocab = []
    with open('ze_format.txt', 'r', encoding='utf-8') as f3:
        lines = f3.readlines()
        for line in lines:
            for word in line:
                z_old_vocab.append(word)
    z_old_len = len(z_old_vocab)
    return z_old_vocab, z_old_len

def get_p_new():
    p_new_vocab = []
    with open('yun_p_new.txt', 'r', encoding='utf-8') as f4:
        lines = f4.readlines()
        for line in lines:
            for word in line:
                p_new_vocab.append(word)
    p_new_len = len(p_new_vocab)
    return p_new_vocab, p_new_len

def get_z_new():
    z_new_vocab = []
    with open('yun_z_new.txt', 'r', encoding='utf-8') as f4:
        lines = f4.readlines()
        for line in lines:
            for word in line:
                z_new_vocab.append(word)
    z_new_len = len(z_new_vocab)
    return z_new_vocab, z_new_len

def get_p():
    p_vocab = []
    with open('yun_p.txt', 'r', encoding='utf-8') as f4:
        lines = f4.readlines()
        for line in lines:
            words = line.split(',')[2]
            for word in words:
                p_vocab.append(word)
    p_len = len(p_vocab)
    return p_vocab, p_len

def get_z():
    z_vocab = []
    with open('yun_z.txt', 'r', encoding='utf-8') as f4:
        lines = f4.readlines()
        for line in lines:
            words = line.split(',')[2]
            for word in words:
                z_vocab.append(word)
    z_len = len(z_vocab)
    return z_vocab, z_len

emb_vocab, emb_len = get_emb()
hzpy_vocab, hzpy_len = get_hzpy()
yun_utf8_vocab, yun_utf8_vocab_li, yun_utf8_len = get_yun_utf8()
p_vocab, p_len = get_p()
z_vocab, z_len = get_z()


print('''
emb_len:{} 
hzpy_len:{} 
yun_utf8_len:{}
p_len:{}
z_len:{}
'''.format(emb_len, hzpy_len, yun_utf8_len, p_len, z_len))

emb_vocab = set(emb_vocab)
hzpy_vocab = set(hzpy_vocab)
yun_utf8_vocab = set(yun_utf8_vocab_li)
p_vocab = set(p_vocab)
z_vocab = set(z_vocab)

oov = emb_vocab - hzpy_vocab
# print(oov)
# print(len(oov))

oov2 = emb_vocab - yun_utf8_vocab
# print(oov2)
# print(len(oov2))

# print(oov2&oov)
oov1 = (emb_vocab - p_vocab) & (emb_vocab - z_vocab) 
# print(oov1)
# print(len(oov1))

oov3 = p_vocab & z_vocab
# print(oov3)
# print(len(oov3))

oov4 = (p_vocab | z_vocab) - yun_utf8_vocab
# print(len(oov4))

    
    
    

