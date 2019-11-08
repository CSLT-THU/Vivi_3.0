# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from constrains import get_next_word
from data_utils import sort_batch_data
from models.Seq2seq_5 import RNN
from word_emb import emb_size, word2id, id2word, emb, word2count, vocab_size, SOS_token, EOS_token, PAD_token, UNK_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERT(nn.Module):
    def __init__(self, model_param):
        super(BERT, self).__init__()
        self.hidden_size = int(model_param['hidden_size'])
        self.input_max_len = int(model_param['input_max_len'])
        self.target_max_len = int(model_param['target_max_len'])
        self.encoder = RNN.EncoderRNN(emb_size, vocab_size,  self.hidden_size).to(device)
        self.decoder = RNN.AttnDecoderRNN(emb_size, vocab_size, self.hidden_size, vocab_size, max_length=self.input_max_len, dropout_p=0.1).to(device)
        
    def forward(self, batch_size, data, criterion,
                teacher_forcing_ratio):
        input_batch, target_batch = data
        
        loss = 0
        
        # encoder
        encoder_outputs = input_batch

        # decoder
        decoder_input = torch.tensor([[SOS_token] * batch_size], device=device).transpose(0, 1)  #
        decoder_hidden = torch.zeros((1, batch_size, self.hidden_size), device=device)
        # decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder

        # target_max_length = max(target_lengths)

        # use_teacher_forcing = True  #
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        sen_len = 8  # 暂时
        sen_num = 5

        for i in range(sen_num):
            for j in range(sen_len): # 算上'/'
                position = torch.tensor([[i, j] for b in range(batch_size)], dtype=torch.float, device=device)
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, position)
                    # decoder_input, decoder_hidden, encoder_outputs_padded)
                di = i * sen_len + j
                target = target_batch[:, di]
                loss += criterion(decoder_output, target)
                if use_teacher_forcing:
                    decoder_input = target.unsqueeze(1)  # Feed the target as the next input
                else:
                    topv, topi = decoder_output.topk(1)  # value 和 id
                    decoder_input = topi.detach()  # detach from history as input

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
            decoded_words = []

            for i in range(sen_num):
                for j in range(sen_len):
                    position = torch.tensor([[i, j]], dtype=torch.float, device=device)
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs_padded, position)
                    if j == 0 and cangtou and i < len(cangtou):
                        top_word = cangtou[i]
                        top_id = torch.LongTensor([word2id.get(top_word, vocab_size - 1)])
                    else:
                        top_id, top_word = get_next_word(decoder_output.data, decoded_words)
                        if top_word == 'N':
                            print('cannot meet requirements')
                            break
                    decoded_words.append(top_word)
                    decoder_input = top_id.reshape((1, 1)).detach()  # detach from history as input

                position = torch.tensor([[i, 7]], dtype=torch.float, device=device)
                tmp_decoder_output, tmp_decoder_hidden, tmp_decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs_padded, position)
                decoder_hidden = tmp_decoder_hidden
                decoder_input = torch.tensor([[2]], device=device)  # '/'作为输入
                
        return decoded_words
        