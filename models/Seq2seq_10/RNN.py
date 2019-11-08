# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

from word_emb import emb_size, word2id, id2word, emb, word2count, vocab_size, SOS_token, EOS_token, PAD_token, UNK_token

with open('resource/lv_emb.json', 'r') as f:
    lv_emb = json.load(f)
    lv_embed_size = len(lv_emb[0])
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# The Seq2Seq Model
# =================
######################################################################
# The Encoder
# -----------

class EncoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size): # input size 没用 就是embed size
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        pretrained_weight = np.array(emb)  # 已有词向量的numpy
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.embedding.weight.requires_grad = False
        
        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, input, input_lengths, hidden):
        embedded = self.embedding(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True) 
        output, hidden = self.gru(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device) # zeros


######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^

class AttnDecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, output_size, max_length=32, dropout_p=0.1):  # 
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        pretrained_weight = np.array(emb)  # 已有词向量的numpy
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.embedding.weight.requires_grad = False

        # attn gru
        self.attn = nn.Linear(self.hidden_size + embed_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + embed_size + 2, self.hidden_size) # Jun2
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, position): 
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[:, 0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)  # unsqueeze插入一个维度

        output = torch.cat((embedded[:, 0], attn_applied[:, 0], position), 1) # Jun2
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.rand(1, 1, self.hidden_size, device=device) # 没用到

class Regularization(nn.Module):
    def __init__(self, train_pattern): # TODO: pattern
        super(Regularization, self).__init__()
        self.w1 = torch.tensor([0.], device=device)
        self.lv_emb = torch.tensor(lv_emb, device=device)
        pattern = [[
            0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0.,
        ]] * 28
        # pattern = [[
        #     0., 1., 0., -1., 0., 1., 0.,
        #     0., -1., 0., 1., 0., -1., 0.,
        #     0., -1., 0., 1., 0., -1., 0.,
        #     0., 1., 0., -1., 0., 1., 0.,
        # ]] * 28
        self.pattern = nn.Parameter(torch.tensor(pattern, device=device, requires_grad=train_pattern))

        self.lv_embedding = nn.Embedding(vocab_size, lv_embed_size)
        lv_pretrained_weight = np.array(lv_emb) 
        self.lv_embedding.weight.data.copy_(torch.from_numpy(lv_pretrained_weight))
        self.lv_embedding.weight.requires_grad = False

    def forward(self, i, j, decoded_words, decoder_output, batch_size):
        n = i * 7 + j
        if n > 0 and j < 7 and i < 4: # n:全诗0-27，除去第一个字1-27 j:不算/ i:不算第五句 
            decoded_words_pad = F.pad(decoded_words, (0, (28-n)), 'constant', 0) # dim=1右边添加28-n个
            dec_emb = self.lv_embedding(decoded_words_pad)
      
            # lv_pred = dec_emb[:, 1].transpose(0, 1) * self.pattern[n, n].data # 不根据历史信息算，纯按模板
            lv_pred = torch.mm(self.pattern[n].unsqueeze(0), dec_emb.transpose(0,1).squeeze(2)) # 根据历史信息
            
            feature1 = torch.mul(self.lv_emb, lv_pred).transpose(0,1) # 4777 * 80
            decoder_output = decoder_output + self.w1 * feature1
        return F.log_softmax(decoder_output, dim=1)
    