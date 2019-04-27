# -*- coding:utf-8 -*-
import json

with open('dicts_word.json', encoding='utf-8') as f1:
    dicts_word = json.load(f1)
with open('dicts_p.json', encoding='utf-8') as f2:
    dicts_p = json.load(f2)
with open('dicts_z.json', encoding='utf-8') as f3:
    dicts_z = json.load(f3)

yun_mode = [
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

yun_li = []
for yun in yun_mode:
    yun_li.extend(yun)
yun_p = []
for dict in dicts_p:
    yun_p.append(dict['yunmu'])
yun_z = []
for dict in dicts_z:
    yun_z.append(dict['yunmu'])
    
yun_li = set(yun_li)
yun_p = set(yun_p)
yun_z = set(yun_z)
print(yun_p - yun_li)
print(yun_z - yun_li)
print(yun_li - yun_p)
print(yun_li - yun_z)

idx2yunmu = {}
for i in range(len(yun_mode)):
        idx2yunmu[i] = yun_mode[i]

def combine_yun(dicts):
    idx2words = {}
    dicts_new = []
    for i in range(len(yun_mode)):
        idx2words[i] = []
    for k, v in idx2yunmu.items():
        for dict in dicts:
            if dict['yunmu'] in v:
                idx2words[k].extend(dict['word'])
        dicts_new.append({'yunmu': v, 'word': idx2words[k]})
    return dicts_new

p = combine_yun(dicts_p)
z = combine_yun(dicts_z)

def save(file, dicts):
    with open(file, 'w', encoding='utf-8') as f1:
        for item in dicts:
            print(item['yunmu'], item['word'])
            for word in item['word']:
                f1.write(word)
            f1.write('\n')
                
save('words_p.txt', p)
save('words_z.txt', z)