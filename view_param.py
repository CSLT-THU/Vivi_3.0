# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
from matplotlib import pyplot as plt  
import importlib

model_name = 'Seq2seq_12'
ckpt_path = 'ckpt/07240456_17_Seq2seq_12_ep=12_loss=158.97.pkl'

checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    # model_param = checkpoint['settings']     # Jun16
model_param = checkpoint['model_param']
model_path = 'models.' + model_name + '.' + model_name
Model = importlib.import_module(model_path)  # 导入模块
model = getattr(Model, model_name)(model_param)  # 反射并实例化
print('model:', model)
model.load_state_dict(checkpoint['model'])
       
params=model.state_dict() 
for k,v in params.items():
    print(k)    #打印网络中的变量名

# pattern
def show_lv_pattern():
    pattern = params['regularization.lv_pattern']
    print('regularization.lv_pattern:')
    for i in range(28):
        print(i, pattern[i].data)

    pattern = pattern.view(28, 4, 7)
    for i in range(28):
        plt.subplot(4, 7, i + 1)
        X = abs(pattern[i])
        plt.axis('off')
        plt.imshow(X)
        # plt.colorbar()
    plt.savefig('eval/heatmap/lv_07240456_12.png')
    # plt.colorbar()
    plt.show()

# yun_pattern
def show_yun_pattern():
    yun_pattern = params['regularization.yun_pattern']
    print('regularization.yun_pattern:')
    for i in range(4):
        print(i, yun_pattern[i].data)
    X = yun_pattern
    plt.axis('off')
    plt.imshow(X)
    plt.colorbar()
    plt.savefig('eval/heatmap/yun_07240456_12.png')
    plt.show() 

show_lv_pattern()
show_yun_pattern()