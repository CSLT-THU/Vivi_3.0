import os
import numpy as np
import re
import sys
import torch
# sys.path.append("../")
# sys.path.append(os.getcwd()+'/../')
from predict import pred
from eval.eval_yunlv import get_rate
from eval.ngrams import get_lm
import json

def test_one(ckpt_path, w1, w2):
    predict_param = {
    'model_name': 'Seq2seq_12',
    'ckpt_path': ckpt_path,
    'cangtou': '',
    'keywords': '',
    'test_set': '',
    'eval_set': 'resource/dataset/test_10k_1k.txt',
    'use_planning': False,
    'bleu_eval': False,
    'poem_type': 'poem7',
    'train_mode': 'kw2poem',
    'note': '',

    'as_train': False,
    'pred_soft': True,
    'template': False,
    'hard_rhyme': True,
    'hard_tone': False,
    'w1': w1,
    'w2': w2,
    }
    
    save_file = pred(predict_param)
    lv_rate, yun_rate = get_rate(save_file, 'result')
    lm = get_lm(save_file)
    return save_file, lv_rate, yun_rate, lm


def test_one_as_train(ckpt_path):
    predict_param = {
        'model_name': 'Seq2seq_12',
        'ckpt_path': ckpt_path,
        'cangtou': '',
        'keywords': '',
        'test_set': '',
        'eval_set': 'resource/dataset/test_10k_1k.txt',
        'use_planning': False,
        'bleu_eval': False,
        'poem_type': 'poem7',
        'train_mode': 'kw2poem',
        'note': '',

        'as_train': True,
        # 'pred_soft': pred_soft,
        # 'template': template,
        'hard_rhyme': True,
        'hard_tone': False,
        # 'w1': w1,
        # 'w2': w2,
    }

    save_file = pred(predict_param)
    lv_rate, yun_rate = get_rate(save_file, 'result')
    lm = get_lm(save_file)
    return save_file, lv_rate, yun_rate, lm
    
li = []

for root,dirs,files in os.walk('ckpt'):
    for dir in dirs:
        if dir.startswith('07270727'):
            path = os.path.join(root,dir)
            # print(path)
            for root2, dirs2, files2 in os.walk(path):
                idx = 0
                loss = 0
                for file in files2:
                    if file == 'loss.npy':
                        dict = np.load(os.path.join(root2,file), allow_pickle=True).item()
                        epoches = dict['plot_epoches']
                        losses = dict['plot_losses']
                        val_losses = dict['plot_val_losses']

                        min_val_loss = min(val_losses)
                        min_idx = val_losses.index(min_val_loss)
                        idx = min_idx
                        loss = losses[idx]
                
                for file in files2:
                    loss = round(loss, 2)
                    re_ep = re.compile(r'_ep='+str(idx+1)+r'_loss='+str(loss))
                    mo = re_ep.search(file)
                    # if mo:
                    if file.startswith('07270727_17_Seq2seq_12_ep=6_loss=150.58'):
                        ckpt_path = os.path.join(root2, file)
                        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
                        train_param = checkpoint['train_param']
                        w1 = train_param['w1']
                        w2 = train_param['w2']

                        if 0:
                            save_file, lv_rate, yun_rate, lm = test_one_as_train(ckpt_path)
                            dic = {
                                        'w1': w1,
                                        'w2': w2,
                                        'lv_rate': lv_rate,
                                        'yun_rate': yun_rate,
                                        'lm': lm,
                                        'ckpt_path': ckpt_path,
                                        'save_file': save_file
                            }
                            li.append(dic)
                            print(dic)
                        
                        if 1:
                            if w2 == 0.:
                                for i in list(np.arange(0., 6.1, 1.)):
                                    save_file, lv_rate, yun_rate, lm = test_one(ckpt_path, float(i), 0.)
                                    dic = {
                                        'w1_pred': float(i),
                                        'w1': w1,
                                        'w2': w2,
                                        'lv_rate': lv_rate,
                                        'yun_rate': yun_rate,
                                        'lm': lm,
                                        'ckpt_path': ckpt_path,
                                        'save_file': save_file
                                    }
                                    li.append(dic)
                                    print(dic)

with open('eval/test_result/07270727_ep6.json', 'w') as f:
    json.dump(li, f)
                       