# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from constrains import get_next_word
from data_utils import sort_batch_data
from models.Transformer.Models import Encoder, Decoder
from word_emb import emb_size, word2id, id2word, emb, word2count, vocab_size, SOS_token, EOS_token, PAD_token, UNK_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Transformer(nn.Module):
    def __init__(
            self, model_param):
        super(Transformer, self).__init__()
        
        n_src_vocab = vocab_size  # src_vocab_size
        n_tgt_vocab = vocab_size  # tgt_vocab_size

        input_max_len = int(model_param['input_max_len']) # ?待定
        target_max_len = int(model_param['target_max_len'])
        tgt_emb_prj_weight_sharing = True if model_param['proj_share_weight']=='True' else False # True
        emb_src_tgt_weight_sharing = True if model_param['embs_share_weight']=='True' else False # True,
        d_k = int(model_param['d_k']) # 64,
        d_v = int(model_param['d_v']) # 64,
        d_model = int(model_param['d_model']) # 200, # embed size
        d_word_vec = int(model_param['d_model']) # 200, # 
        d_inner = int(model_param['d_inner_hid']) # 2048,
        n_layers = int(model_param['n_layers']) # 6,
        n_head = int(model_param['n_head']) # 8,
        dropout = float(model_param['dropout']) # 0.1

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=input_max_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=target_max_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        
        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        # if tgt_emb_prj_weight_sharing:
        #     # Share the weight matrix between target word embedding & the final logit dense layer
        #     self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
        #     self.x_logit_scale = (d_model ** -0.5)
        # else:
        #     self.x_logit_scale = 1.
        # 
        # if emb_src_tgt_weight_sharing:
        #     # Share the weight matrix between source & target word embeddings
        #     assert n_src_vocab == n_tgt_vocab, \
        #         "To share word embedding table, the vocabulary size of src/tgt shall be the same."
        #     self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

        # test
        self.step= 0 
        
    def forward(self, batch_size, data, criterion,
                teacher_forcing_ratio):
        src_seq, src_pos, tgt_seq, tgt_pos = data

        # TODO: 没有用scheduled sampling
        enc_output, *_ = self.encoder(src_seq, src_pos)
        enc_output = torch.ones_like(enc_output) # Jun24 test**************
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) # Jun21
        seq_logit = F.log_softmax(seq_logit, dim=1) 
        # seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale 

        pred = seq_logit.contiguous().view(-1, seq_logit.size(2)) # add contiguous()
        gold = tgt_seq.contiguous().view(-1)

        loss = criterion(pred, gold)
        loss = loss * len(gold) / batch_size
        
        # test
        self.step += 1
        if self.step % 100 == 0:
            print('step:', self.step)
            print('src:', [id2word[str(x.data.cpu().numpy())] for x in src_seq[0]])
            print('tgt:', [id2word[str(x.data.cpu().numpy())] for x in tgt_seq[0]])
            # print(seq_logit[0], tgt_seq[0])
            sen_loss = criterion(seq_logit[0], tgt_seq[0])
            print('sen_loss:', sen_loss)
            print('loss:', loss)
            
            li = [x.data.cpu().numpy() for x in seq_logit[0]]
            for j in range(6):
                max_w = []
                for i in range(40):
                    max_idx = np.argmax(li[i])
                    max_w.append(max_idx)
                    li[i][max_idx] = -1000000
                print('pred', j, [id2word[str(w)] for w in max_w])
            
        return loss
    
    def predict(self, data, cangtou, predict_param):
        with torch.no_grad():
            src_seq, src_pos = data

            # test
            # print('src:', [id2word[str(x.to(device).numpy())] for x in src_seq[0]])

            enc_output, *_ = self.encoder(src_seq, src_pos)
            
            # dec_seq = torch.tensor([[2]], device=device) # (beam size, decoded len) # 不要用SOS_token, 0会导致decoder_output为nan. 此处使用/
            # dec_pos = torch.tensor([[1]], device=device)
            
            sen_len = 7
            sen_num = 4
            decoded_words = ['/']
            rt_decoded_words = []
            
            for i in range(sen_num):
                for j in range(sen_len):
                    dec_seq = torch.tensor([[word2id[word] for word in decoded_words]], dtype=torch.long, device=device)
                    dec_pos = torch.tensor([[k + 1 for k in range(len(decoded_words))]], dtype=torch.long, device=device)
                    dec_output, *_ = self.decoder(dec_seq, dec_pos, src_seq, enc_output)
                    dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                    word_prob = F.log_softmax(self.tgt_word_prj(dec_output), dim=1) # (1, beam size, 4777)
                    top_id, top_word = get_next_word(word_prob.data, rt_decoded_words) # 应该是(1, 4777)
                    if top_word == 'N':
                        print('cannot meet requirements')
                        break
                    decoded_words.append(top_word)
                    rt_decoded_words.append(top_word)
                if not i==sen_num:
                    decoded_words.append('/')
                
            # test
            # print('pred:', [id2word[str(x.to(device).numpy())] for x in torch.argmax(decoded_words, dim=1)])

        return rt_decoded_words