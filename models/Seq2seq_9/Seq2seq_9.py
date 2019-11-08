# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
from torch.autograd import Variable

from constrains import get_next_word
from get_feature import get_feature
from data_utils import sort_batch_data
from models.Seq2seq_9 import RNN

from word_emb import emb_size, word2id, id2word, emb, word2count, vocab_size, SOS_token, EOS_token, PAD_token, UNK_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('resource/word_dict.json', 'r', encoding='utf-8') as f:
    word_dict = json.load(f)

def get_self_attn(decoded_words, fix_len, batch_size):
    if decoded_words.shape[1] < fix_len:
        self_attn_input = decoded_words
        self_attn_pos = torch.arange(1, decoded_words.shape[1] + 1, dtype=torch.long, device=device)
        self_attn_pos = self_attn_pos.unsqueeze(0).repeat(batch_size, 1)
        # self_attn_pos = self_attn_pos.unsqueeze(0).repeat(batch_size, 0)
    else:
        self_attn_input = decoded_words[:, -fix_len:]
        self_attn_pos = torch.arange(decoded_words.shape[1] - self.fix_len + 1, decoded_words.shape[1] + 1,
                                     dtype=torch.long, device=device)
        self_attn_pos = self_attn_pos.unsqueeze(0).repeat(batch_size, 1)
    return self_attn_input, self_attn_pos


class Seq2seq_9(nn.Module):
    def __init__(self, model_param):
        super(Seq2seq_9, self).__init__()
        # yunlv_emb
        yunlv_emb_size = 18
        # self attn param
        n_head = int(model_param['n_head'])
        d_model = int(model_param['d_model'])
        d_k = int(model_param['d_k'])
        d_v = int(model_param['d_v'])
        # decoded words length
        self.fix_len = int(model_param['fix_len'])
        self.last_y = True if model_param['last_y'] == 'True' else False
        # other param
        hidden_size = int(model_param['hidden_size'])
        self.input_max_len = int(model_param['input_max_len'])
        self.target_max_len = int(model_param['target_max_len'])
        self.encoder = RNN.EncoderRNN(emb_size, vocab_size,  hidden_size).to(device)
        self.decoder = RNN.AttnDecoderRNN(emb_size, yunlv_emb_size, vocab_size, hidden_size, vocab_size, n_head, 
                                          d_model, d_k, d_v, max_length=self.input_max_len, 
                                          tgt_max_len=self.target_max_len, dropout_p=0.1).to(device)
        # word dict
        self.word_dict = word_dict
        # rhyme loss
        rhyme_loss = True if model_param['rhyme_loss'] == 'True' else False
        self.generator = RNN.Generator(rhyme_loss)
         
        # test
        # self.step = 0
        
    def forward(self, batch_size, data, criterion,
                teacher_forcing_ratio):
        # input_batch, input_pos, input_lengths, target_batch, target_pos, target_lengths = data
        # input_lengths, [input_batch, input_pos, target_batch, target_pos, target_lengths] = sort_batch_data2(
        #     input_lengths, [input_batch, input_pos, target_batch, target_pos, target_lengths])
        input_batch, input_lengths, target_batch, target_lengths = data
        input_batch, input_lengths, target_batch, target_lengths = sort_batch_data(
            input_batch, input_lengths, target_batch, target_lengths)
        
        loss = 0
        
        # encoder
        encoder_hidden = self.encoder.initHidden(batch_size)  #
        encoder_outputs, encoder_hidden = self.encoder(input_batch, input_lengths, encoder_hidden)
        
        ################### test same encoder output ######################
        # encoder_outputs = torch.ones_like(encoder_outputs)  # Jun24 
        
        # 将encoder_outputs padding至INPUT_MAX_LENGTH 因为attention中已经固定此维度大小为INPUT_MAX_LENGTH
        encoder_outputs_padded = torch.zeros(batch_size, self.input_max_len, self.encoder.hidden_size,
                                             device=device)
        for b in range(batch_size):
            for ei in range(input_lengths[b]):
                encoder_outputs_padded[b, ei] = encoder_outputs[b, ei]

        # decoder
        decoder_input = torch.tensor([[SOS_token] * batch_size], device=device).transpose(0, 1)  #
        decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

        # use_teacher_forcing = True  #
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        sen_len = 8  # 暂时
        sen_num = 5 # Jun24 原来是5
        
        decoded_words = torch.tensor([[SOS_token] * batch_size], device=device).transpose(0, 1) # Jun25
        
        # test
        outputs = []

        for i in range(sen_num):
            for j in range(sen_len): # 算上'/'
                position = torch.tensor([[i, j] for b in range(batch_size)], dtype=torch.float, device=device)
                self_attn_input, self_attn_pos = get_self_attn(decoded_words, self.fix_len, batch_size)
                
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs_padded, position, self_attn_input, self_attn_pos, self.last_y)

                di = i * sen_len + j
                target = target_batch[:, di]

                decoder_output = self.generator(i, j, decoded_words, decoder_output, batch_size, target)
                
                # if self.rhyme_loss:
                #     if j < 7 and i < 4:
                #         feature1, feature2 = get_feature(decoded_words[:, 1:].cpu().numpy().tolist(), i, j, batch_size, target)
                #         decoder_output = decoder_output + self.w1 * torch.from_numpy(feature1).to(
                #             device) + self.w2 * torch.from_numpy(feature2).to(device)
                # 
                # decoder_output = F.log_softmax(decoder_output, dim=1)

                loss += criterion(decoder_output, target)
                     
                if use_teacher_forcing:
                    decoder_input = target.unsqueeze(1)  # Feed the target as the next input
                else:
                    topv, topi = decoder_output.topk(1)  # value 和 id
                    decoder_input = topi.detach()  # detach from history as input
                
                decoded_words = torch.cat((decoded_words, decoder_input), 1) # Jun25
                
                # test
                topv, topi = decoder_output.topk(6)
                outputs.append(topi[0].detach())
        # test
        # self.step += 1
        # if self.step % 100 == 0:
        #     print('step:', self.step)
        #     print('source:', [id2word[str(x.data.cpu().numpy())] for x in input_batch[0]])
        #     print('target:', [id2word[str(x.data.cpu().numpy())] for x in target_batch[0]])
        #     for i in range(6):
        #         print('pred:', i, [id2word[str(x[i].cpu().data.numpy()).strip('[').strip(']')] for x in outputs])
        
        return loss
    
    def predict(self, data, cangtou, predict_param):
        hard_rhyme = predict_param['hard_rhyme']
        with torch.no_grad():
            encoder_hidden = self.encoder.initHidden(1)

            input_sentence, input_length = data
            encoder_outputs, encoder_hidden = self.encoder(input_sentence, [input_length], encoder_hidden)

            # 将encoder_outputs padding至INPUT_MAX_LENGTH 因为attention中已经固定此维度大小为INPUT_MAX_LENGTH
            encoder_outputs_padded = torch.zeros(1, self.input_max_len, self.encoder.hidden_size,
                                                 device=device)
            for b in range(1):
                for ei in range(input_length):
                    encoder_outputs_padded[b, ei] = encoder_outputs[b, ei]

            decoder_input = torch.tensor([[SOS_token]], device=device)  # 第一个input是START
            decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

            decoded_words = torch.tensor([[SOS_token] * 1], device=device).transpose(0, 1)  # Jun25
            
            sen_len = 7 # 暂时
            sen_num = 4
            rt_decoded_words = []

            for i in range(sen_num):
                for j in range(sen_len):
                    position = torch.tensor([[i, j]], dtype=torch.float, device=device)
                    self_attn_input, self_attn_pos = get_self_attn(decoded_words, self.fix_len, 1)
                    
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs_padded, position, self_attn_input, self_attn_pos,
                        self.last_y)

                    decoder_output = F.log_softmax(decoder_output, dim=1)
                    
                    if j == 0 and cangtou and i < len(cangtou):
                        top_word = cangtou[i]
                        top_id = torch.LongTensor([word2id.get(top_word, vocab_size - 1)])
                    else:
                        top_id, top_word = get_next_word(decoder_output.data, rt_decoded_words, hard_rhyme=hard_rhyme)
                        if top_word == 'N':
                            print('cannot meet requirements')
                            break
                    rt_decoded_words.append(top_word)
                    decoder_input = top_id.reshape((1, 1)).detach()  # detach from history as input
                    decoded_words = torch.cat((decoded_words, decoder_input), 1)  # Jun25
                
                position = torch.tensor([[i, 7]], dtype=torch.float, device=device)
                self_attn_input, self_attn_pos = get_self_attn(decoded_words, self.fix_len, 1)
                
                tmp_decoder_output, tmp_decoder_hidden, tmp_decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs_padded, position,  self_attn_input, self_attn_pos, self.last_y)
                decoder_hidden = tmp_decoder_hidden
                decoder_input = torch.tensor([[2]], device=device)  # '/'作为输入
                
        return rt_decoded_words
    