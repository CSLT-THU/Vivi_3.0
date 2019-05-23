# -*- coding: utf-8 -*-
from sklearn import model_selection

# WARNING: 没事不要随便运行，否则数据集全打乱了，bleu文件全要替换，模型也不能从ckpt继续训了。

dataset_file = 'poem_480.txt'
train_set_file = 'train_480.txt'
test_set_file = 'test_480.txt'
test_set_source_file = 'test_480_source.txt'
test_set_target_file = 'test_480_target.txt'

with open(dataset_file, 'r', encoding = 'utf-8') as f:
    lines = f.readlines()
    train_set, test_set = model_selection.train_test_split(lines, test_size=0.1, random_state=None)

with open(train_set_file, 'w', encoding = 'utf-8') as f1:
    for line in train_set:
        f1.write(line)
        
with open(test_set_file, 'w', encoding = 'utf-8') as f2:
    for line in test_set:
        f2.write(line)

with open(test_set_source_file, 'w', encoding = 'utf-8') as f3:
    for line in test_set:
        source = line.split('==')[0]
        f3.write(source + '\n')

with open(test_set_target_file, 'w', encoding = 'utf-8') as f4:
    for line in test_set:
        target = line.split('==')[1]
        f4.write(target)