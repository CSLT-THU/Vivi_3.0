# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        self.input_size = 
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        self.cell = nn.RNNCell(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, position): 
        
        return output, hidden

    def initHidden(self):
        return torch.rand(1, 1, self.hidden_size, device=device) # 没用到
