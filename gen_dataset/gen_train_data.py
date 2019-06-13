import json
import random
import time
# randomly sample keywords from the poem

with open('../resource/sxhy_dict.txt', 'r', encoding='utf-8') as f0:
    l = f0.readline()
    sxhy_li = l.split(' ')  # 34528
    
def segment(sentence):        
    toks = []
    for slot in [2, 3]:
        idx = 0
        while idx + slot <= len(sentence):
            # Cut 2 or 3 chars each time.
            if sentence[idx: idx + slot] in sxhy_li:
                if sentence[idx: idx + slot] not in toks:
                    toks.append(sentence[idx: idx + slot])
            idx += 1
    for word in sentence:
        flag = 0
        for tok in toks:
            if word in tok:
                flag = 1
        if flag == 0:
            toks.append(word)
    return toks

def build_kw_dict(file, kw_file):
    kw_dict = {}
    cnt = 0
    start = time.time()
    with open(file, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            cnt += 1
            if cnt % 100 == 0:
                print('build kw dict:', cnt, 'time:', round((time.time()-start) / 60, 1))
            poem = l.split('==')[1]
            poem = poem.replace('\n', '')
            poem = poem.replace(' ', '')
            lines = poem.split('\t')
            for line in lines:
                kws = segment(line)
                # print(line, kws)
                kw_dict[line] = kws
        # print(len(kw_dict))

    with open(kw_file, 'w', encoding='utf-8') as f1:
        json.dump(kw_dict, f1)     
        print('kw dict built')
        
def sample_kw(kws, num):
    kw_samples = []
    # kw_samples = random.sample(kws, 4) # 无重复
    for i in range(num): # 有重复
        kw_samples.append(kws[random.randint(0, len(kws)-1)])
    return kw_samples

def rand_kw(num, file, kw_file, new_file):
    content = ''
    with open(kw_file, 'r', encoding='utf-8') as f1:
        kw_dict = json.load(f1)    
    with open(file, 'r', encoding='utf-8') as f:
        cnt = 0
        for l in f.readlines():
            cnt += 1
            if cnt % 100 == 0:
                print('sample kw:', cnt)
            poem_ = l.split('==')[1]
            poem = poem_.replace('\n', '')
            poem = poem.replace(' ', '')
            lines = poem.split('\t')
            poem_kw_samples = []
            for line in lines:
                kws = kw_dict[line]
                kw_samples = sample_kw(kws, num)
                # print(cnt, kws, kw_samples)
                poem_kw_samples.append(kw_samples)
            # print(poem_kw_samples)
            for i in range(num):
                for j in range(4):
                    source = ''
                    for word in poem_kw_samples[j][i]:
                        source += word + ' '
                    source = source[0:-1]
                    content += source 
                    if j != 3:
                        content += ' - '
                content += '==' + poem_
            # print(cnt, poem_kw_samples, [kw_dict[line] for line in lines], poem)    
    with open(new_file, 'w', encoding='utf-8') as f2:
        f2.write(content)

build_kw_dict('../resource/dataset/poem_1031k_theme.txt', '../resource/dataset/kw_1031k.json')
rand_kw(4, '../resource/dataset/poem_1031k_theme.txt', '../resource/dataset/kw_1031k.json', 
        '../resource/dataset/poem_1031k_4x.txt')                