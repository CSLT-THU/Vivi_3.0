# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from constrains import get_next_word
from get_feature import get_feature
from data_utils import sort_batch_data
from models.Seq2seq_new import RNN
from word_emb import emb_size, word2id, id2word, emb, word2count, vocab_size, SOS_token, EOS_token, PAD_token, UNK_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Seq2seq_new(nn.Module):
    def __init__(self, model_param):
        super(Seq2seq_new, self).__init__()
        hidden_size = int(model_param['hidden_size'])
        self.input_max_len = int(model_param['input_max_len'])
        self.target_max_len = int(model_param['target_max_len'])
        self.encoder = RNN.EncoderRNN(emb_size, vocab_size,  hidden_size).to(device)
        self.decoder = RNN.AttnDecoderRNN(emb_size, vocab_size, hidden_size, vocab_size, max_length=self.input_max_len, dropout_p=0.1).to(device)
        self.linears = RNN.Linears()
        
    def forward(self, batch_size, data, criterion,
                teacher_forcing_ratio):
        input_batch, input_lengths, target_batch, target_lengths = data
        input_batch, input_lengths, target_batch, target_lengths = sort_batch_data(
            input_batch, input_lengths, target_batch, target_lengths)
        
        loss = 0
        
        # encoder
        encoder_hidden = self.encoder.initHidden(batch_size)  #
        encoder_outputs, encoder_hidden = self.encoder(input_batch, input_lengths, encoder_hidden)
        # 将encoder_outputs padding至INPUT_MAX_LENGTH 因为attention中已经固定此维度大小为INPUT_MAX_LENGTH
        encoder_outputs_padded = torch.zeros(batch_size, self.input_max_len, self.encoder.hidden_size,
                                             device=device)
        for b in range(batch_size):
            for ei in range(input_lengths[b]):
                encoder_outputs_padded[b, ei] = encoder_outputs[b, ei]

        # decoder
        decoder_input = torch.tensor([[SOS_token] * batch_size], device=device).transpose(0, 1)  #
        decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

        sen_len = 7  # 暂时
        sen_num = 4
        decoded_words = [[] for n in range(batch_size)]

        for i in range(sen_num):
            for j in range(sen_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs_padded)
                
                feature1, feature2 = get_feature(decoded_words, i, j, batch_size)
                feature1 = torch.tensor(feature1).to(device)
                feature2 = torch.tensor(feature2).to(device)
                decoder_output = self.linears(decoder_output, feature1, feature2)

                target = target_batch[:, i*(sen_num+1)+j] #####
                loss += criterion(decoder_output, target)
            
                topv, topi = decoder_output.topk(1)  # value 和 id (80,1)
                decoder_input = topi.detach()  # detach from history as input
                
                for b in range(batch_size):
                    top_word = id2word[str(topi[b].item())] # 一个batch
                    decoded_words[b].append(top_word)

            tmp_decoder_output, tmp_decoder_hidden, tmp_decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs_padded)
            decoder_hidden = tmp_decoder_hidden
            decoder_input = torch.tensor([[2] for n in range(batch_size)], device=device)  # '/'作为输入

        return loss
    
    def predict(self, data, cangtou, predict_param):
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

            sen_len = 7 # 暂时
            sen_num = 4

            decoded_words = [[]]
            for i in range(sen_num):
                for j in range(sen_len):
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs_padded)

                    feature1, feature2 = get_feature(decoded_words, i, j, 1) # 训练也用此函数 需要有batch size的维度
                    feature1 = torch.tensor(feature1).to(device)
                    feature2 = torch.tensor(feature2).to(device)
                    decoder_output = self.linears(decoder_output, feature1, feature2)

                    if j == 0 and cangtou and i < len(cangtou):
                        top_word = cangtou[i]
                        top_id = torch.LongTensor([word2id.get(top_word, vocab_size - 1)])
                        decoder_input = top_id.reshape((1, 1)).detach()
                    else:
                        topv, topi = decoder_output.topk(1)  # value 和 id (1,1)
                        decoder_input = topi.detach()  # detach from history as input
                        top_word = id2word[str(topi[0].item())]  # 一个batch
                        if top_word == 'N':
                            print('cannot meet requirements')
                            break
                    decoded_words[0].append(top_word)
                        
                tmp_decoder_output, tmp_decoder_hidden, tmp_decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs_padded)
                decoder_hidden = tmp_decoder_hidden
                decoder_input = torch.tensor([[2] for n in range(1)], device=device)  # '/'作为输入
                
        return decoded_words
        