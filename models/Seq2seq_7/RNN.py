# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.Seq2seq_7.modules import LSTMON, LSTMONCell

from word_emb import emb_size, word2id, id2word, emb, word2count, vocab_size, SOS_token, EOS_token, PAD_token, UNK_token

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

        self.rnn = LSTMON(embed_size, hidden_size) # Jun10
        # self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, input, input_lengths, hidden):
        embedded = self.embedding(input)
        # packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True) 
        h, c = self.rnn(embedded, hidden) # 在ONLSTM中初始化hidden，不在这里定义
        output = h
        # output, hidden = self.gru(packed, hidden)
        # output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, (h[:,-1], c[:,-1]) # lstm的output就是h h和c均为输入长度 只取最后一个

    def initHidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.hidden_size, device=device) # 是否有必要Variable?
        # zeros = Variable(torch.zeros(batch_size, self.hidden_size))
        initial_states = [(zeros, zeros)]
        return initial_states
        # return (torch.zeros(1, batch_size, self.hidden_size, device=device),
        #         torch.zeros(1, batch_size, self.hidden_size, device=device)) # zeros
    # def initHidden(self, batch_size):
    #     return torch.zeros(1, batch_size, self.hidden_size, device=device) # zeros


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
        self.rnn = LSTMON(hidden_size, hidden_size)  # Jun10
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, position): 
        h, c = hidden
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[:, 0], h), 1)), dim=1)
        # attn_weights = F.softmax(
            # self.attn(torch.cat((embedded[:, 0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)  # unsqueeze插入一个维度

        output = torch.cat((embedded[:, 0], attn_applied[:, 0], position), 1) # Jun2
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        
        output = output.transpose(0,1) 
        h, c = self.rnn(output, [hidden])
        output = h
        # output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[:, 0]), dim=1)
        
        return output, (h[:,-1], c[:,-1]), attn_weights

    # def initHidden(self):
    #     return torch.rand(1, 1, self.hidden_size, device=device) # 没用到
