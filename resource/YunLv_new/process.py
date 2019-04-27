# -*- coding: utf8 -*-
import json
import copy

def get_emb():
    with open("../embedding/word2id.json", 'r') as f0:
        dic = json.load(f0)
        emb_vocab = list(dic.keys())
    return emb_vocab

def get_pzsy(emb_vocab):
    cnt = 0
    dicts = []
    exception = []
    with open('yun_utf8.txt', 'r', encoding='utf-8') as f0:
        for line in f0.readlines():
            line = line.strip('\n')
            line = line.strip(' ')
            line = line.replace('  ', ' ')
            if len(line.split(' ')) > 1:
                word = line.split(' ')[0]
                pz = line.split(' ')[1]
                if word in emb_vocab:
                    dicts.append({'word': word, 'pz': pz, 'sheng': '', 'yun': ''})
                    cnt += 1
                    print(cnt)
            else:
                print('exception occurs')
                exception.append(line.split(' '))
                continue
        print('exception in pz file:', exception)
    
    dicts2 = copy.copy(dicts) # 对dicts做append不会影响dicts2 但是修改元素值会影响
    with open('hzpy.txt', 'r', encoding='utf-8-sig') as f1:
        lines = f1.readlines()
        for line in lines:
            flag = 0
            line = line.strip('\n')
            line = line.split(',')
            word = line[0]
            sheng = line[2]
            yun = line[3]
            if word in emb_vocab:
                for dict in dicts:
                    pz = dict['pz']
                    if word == dict['word']: 
                        if dict['yun'] == '' and dict['sheng'] == '': 
                            dict['sheng'] = sheng
                            dict['yun'] = yun
                        else: # 解决多音字覆盖问题，当成不同字处理
                            dicts2.append({'word': word, 'pz': pz, 'sheng': sheng, 'yun': yun}) 
                        flag = 1
                    else:
                        pass
                if flag == 0:
                    dicts2.append({'word': word, 'pz': '', 'sheng': sheng, 'yun': yun})
                    cnt += 1
                    print(cnt, 'word with yun no pz')
    return dicts2

def eni(dicts):
    for dict in dicts:
        if dict['yun'] == 'i':
            if dict['sheng'] in ['z', 'c', 's']:
                dict['yun'] = 'i1'
            elif dict['sheng'] in ['zh', 'ch', 'sh', 'r']:
                dict['yun'] = 'i2'
        if dict['yun'] == 'e':
            if dict['sheng'] == 'y':
                dict['yun'] = 'e1'
    return dicts

def classify_by_yun(dicts):
    p_yunmu = []
    z_yunmu = []
    p_words = []
    z_words = []
    for dict in dicts:
        word = dict['word']
        pz = dict['pz']
        yun = dict['yun']
        if pz == 'p':
            if yun not in p_yunmu:
                p_yunmu.append(yun)
                p_words.append({'yunmu': yun, 'word': [word]})
            else:
                for dic in p_words:
                    if yun == dic['yunmu']:
                        dic['word'].append(word)
        elif pz == 'z':
            if yun not in z_yunmu:
                z_yunmu.append(yun)
                z_words.append({'yunmu': yun, 'word': [word]})
            else:
                for dic in z_words:
                    if yun == dic['yunmu']:
                        dic['word'].append(word)
    return p_words, z_words

def count3(dicts):
    all_words = []
    no_yun = []
    no_pz = []
    p_words = []
    z_words = []
    pz_error = []
    for dict in dicts:
        all_words.append(dict['word'])
        if dict['yun'] == '':
            no_yun.append(dict['word'])
        if dict['pz'] == '':
            no_pz.append(dict['word'])
        elif dict['pz'] == 'p':
            p_words.append(dict['word'])
        elif dict['pz'] == 'z':
            z_words.append(dict['word'])
        else: 
            pz_error.append(dict)
    return all_words, no_yun, no_pz, p_words, z_words, pz_error

emb_vocab = get_emb()
# yun_utf8_vocab, p_words, z_words = get_yun_utf8()
dicts = get_pzsy(emb_vocab)
dicts = eni(dicts) # 区分ye和e，区分zi ci si，zhi chi shi和普通i
p_dicts, z_dicts = classify_by_yun(dicts)

with open('dicts_word.json','w', encoding='utf-8') as f0:
    json.dump(dicts,f0, ensure_ascii=False)
with open('dicts_p.json','w', encoding='utf-8') as f1:
    json.dump(p_dicts,f1, ensure_ascii=False)
with open('dicts_z.json','w', encoding='utf-8') as f2:
    json.dump(z_dicts,f2, ensure_ascii=False)

all_words, no_yun, no_pz, p_words, z_words, pz_error = count3(dicts)

all_words_set = set(all_words)
emb_vocab_set = set(emb_vocab)

print('dict len:', len(dicts))
print('all words len:', len(all_words))
print('all words set len:', len(all_words_set))
print('oov:', emb_vocab_set - all_words_set)
print('no yun:', no_yun)
print('no pz:', no_pz)
print('pz error:', pz_error)

print('''
p_words
yunmu count:{} 
words count:{} 
'''.format(len(p_dicts), len(p_words)))
for dic in p_dicts:
    print(dic['yunmu'], end=', ')
print('\n')
for dic in p_dicts:
    print(dic)

print('''
z_words
yunmu count:{} 
words count:{} 
'''.format(len(z_dicts), len(z_words)))
for dic in z_dicts:
    print(dic['yunmu'], end=', ')
print('\n')
for dic in z_dicts:
    print(dic)
