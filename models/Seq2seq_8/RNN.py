# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from get_feature import get_feature
from models.Transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward # Jun25 待改
from word_emb import emb_size, word2id, id2word, emb, word2count, vocab_size, SOS_token, EOS_token, PAD_token, UNK_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(PAD_token).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD_token)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class EncoderRNN(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size): # input size 没用 就是embed size
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, input_size)
        pretrained_weight = np.array(emb)  # 已有词向量的numpy
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.embedding.weight.requires_grad = False
        
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, input_lengths, hidden):
        embedded = self.embedding(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True) 
        output, hidden = self.gru(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device) # zeros


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, output_size, n_head, d_model, d_k, d_v, max_length=32, tgt_max_len=40, dropout_p=0.1):  # 
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.tgt_max_len = tgt_max_len # Jul1

        # embedding
        self.embedding = nn.Embedding(vocab_size, input_size)
        pretrained_weight = np.array(emb)  # 已有词向量的numpy
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.embedding.weight.requires_grad = False

        self.dropout = nn.Dropout(self.dropout_p)
        
        # Add self-attn # Jun25
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=self.dropout_p)
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.tgt_max_len, input_size, padding_idx=0), freeze=True)
        
        # attn
        self.attn = nn.Linear(self.hidden_size + input_size, self.max_length)
        # TODO attn_combine是否有必要? Jul1
        self.attn_combine = nn.Linear(self.hidden_size + input_size + 2 + d_model, self.hidden_size)  # Jun25 
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
        # add yunlv
        # self.w1 = Variable(torch.tensor([1.], device=device), requires_grad=True)  #
        # self.w2 = Variable(torch.tensor([1.], device=device), requires_grad=True)

    def forward(self, input, hidden, encoder_outputs, position, dec_seq, dec_pos, last_y): 
        # emb
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        
        # attn
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[:, 0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)  # unsqueeze插入一个维度

        # self-attn
        # masks
        non_pad_mask = get_non_pad_mask(dec_seq)
        slf_attn_mask_subseq = get_subsequent_mask(dec_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=dec_seq, seq_q=dec_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        # emb
        dec_input = self.embedding(dec_seq) + self.position_enc(dec_pos) # Jul1
        # dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask
        dec_output = dec_output[:, -1, :] # 应该是[80, 200] ?
        
        #################### test ####################
        # dec_output = torch.zeros_like(dec_output)

        # combine
        output = torch.cat((embedded[:, 0], attn_applied[:, 0], position, dec_output), 1)  # Jun2
        # output = torch.cat((embedded[:, 0], attn_applied[:, 0], position), 1) # Jun2
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        
        # gru
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        
        # add yunlv
        # rhyme_loss, i, j, decoded_words, batch_size = rhyme_data
        # if rhyme_loss:
        #     if j < 7 and i < 4:
        #         feature1, feature2 = get_feature(decoded_words[:, 1:].cpu().numpy().tolist(), i, j, batch_size)
        #         output = output + self.w1 * torch.from_numpy(feature1).to(device) + \
        #                  self.w2 * torch.from_numpy(feature2).to(device)

        # output = F.log_softmax(output, dim=1)
        
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.rand(1, 1, self.hidden_size, device=device) # 没用到


class Generator(nn.Module):
    def __init__(self, rhyme_loss):
        super(Generator, self).__init__()
        self.rhyme_loss = rhyme_loss
        self.w1 = torch.tensor([1.], device=device)
        self.w2 = torch.tensor([1.], device=device)

    def forward(self, i, j, decoded_words, decoder_output, batch_size, target):
        if self.rhyme_loss:
            if j < 7 and i < 4:
                feature1, feature2 = get_feature(decoded_words[:, 1:].cpu().numpy().tolist(), i, j, batch_size, target)
                decoder_output = decoder_output + self.w1 * torch.from_numpy(feature1).to(
                    device) + self.w2 * torch.from_numpy(feature2).to(device)
        return F.log_softmax(decoder_output, dim=1)