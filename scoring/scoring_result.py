# -*- coding: utf-8 -*-
import json 
import pandas as pd
import os

# config
name = '1031k_0831_3/'
models = ['base', 'base+rule', 'soft', 'soft+rule', 'jiuge']
n_poems = len(models)
avg_scores = {model: [0, 0, 0, 0, 0] for model in models}

cnt = 0
for root,dirs,files in os.walk(name):
    for dir in dirs:
        score_file = root + dir + '/All_Data_Readable.csv'
        if os.path.exists(score_file):
            cnt += 1
            data = pd.read_csv(score_file, encoding='gbk')
            
            idx_file = root + dir + '/idx.json'
            with open(idx_file, 'r', encoding='utf-8') as f:
                idx_li = json.load(f)
            n_questions = len(idx_li)
            
            scores = {model: [0, 0, 0, 0] for model in models}  # 每个模型4个指标           
    
            for i in range(n_questions):
                for j in range(n_poems):
                    score_for_one_poem = []
                    for k in range(4):
                        # print(str(i)+' '+str(j)+' '+str(k)+':')
                        # print(float(data.iloc[0, 12+(j*4):16+(j*4)][k]))
                        # score_for_one_poem.append(float(data.iloc[0, 12+(j*4+1):16+(j*4+1)][k]))
                        score_for_one_poem.append(float(data.iloc[0, 14 + (j * 4 + 1):18 + (j * 4 + 1)][k]))
                    model = models[idx_li[i][j]-1]
                    scores[model] = [score_for_one_poem[s] + scores[model][s] for s in range(4)]

            for model in models:
                avg = 0
                scores[model] = [x / n_questions for x in scores[model]]
                for score in scores[model]:
                    avg += score
                scores[model].append(avg / 4)
            print(dir)
            print(scores)
            
            for model in models:
                for i in range(5):
                    avg_scores[model][i] += scores[model][i]

for model in models:
    for i in range(5):
        avg_scores[model][i] = round(avg_scores[model][i] / cnt, 3)
print('avg:')
print(avg_scores)
                