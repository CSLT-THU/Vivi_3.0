# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
from torch.utils.data.dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SRC_MAX_LENGTH = 32
TGT_MAX_LENGTH = 41


class PoetryData(Dataset):
    def __init__(self, data, src_max_len=SRC_MAX_LENGTH, tgt_max_len=TGT_MAX_LENGTH, test=False):
        '''
        when chunk size = 120, evenly divide; = 259, leave one out
        most poetries have length around 40 - 80
        data is nested list of word idx 嵌套列表 
        '''
        # assert any(isinstance(i, tuple) for i in data)
        
        self.test = test
        self.lens_1 = len(data[0])
        self.lens_2 = len(data[1])
        self.lens = len(data)

        if test:
            source_seqs_1, source_seqs_2 = data
            # target_seqs = [[0] for i in range(max(len(source_seqs_1),len(source_seqs_2)))]
            self.source_lens_1 = torch.LongTensor([min(len(x), src_max_len) for x in source_seqs_1])
            self.source_lens_2 = torch.LongTensor([min(len(x), src_max_len) for x in source_seqs_2])
            src_pad_len_1 = min(max(self.source_lens_1), src_max_len)
            src_pad_len_2 = min(max(self.source_lens_2), src_max_len)
            self.source_1 = torch.zeros((self.lens_1, src_pad_len_1)).long()
            self.source_2 = torch.zeros((self.lens_2, src_pad_len_2)).long()
            for i in range(self.lens_1):
                TL = min(self.source_lens_1[i], src_max_len)
                self.source_1[i, :TL] = torch.LongTensor(source_seqs_1[i][:TL])
            for i in range(self.lens_2):
                TL = min(self.source_lens_2[i], src_max_len)
                self.source_2[i, :TL] = torch.LongTensor(source_seqs_2[i][:TL])
            self.source_1 = self.source_1.to(device)
            self.source_2 = self.source_2.to(device)
            self.source_lens_1 = self.source_lens_1.to(device)
            self.source_lens_2 = self.source_lens_2.to(device)
        
        else:
            source_seqs = [i[0] for i in data]
            target_seqs = [i[1] for i in data]

            self.source_lens = torch.LongTensor([min(len(x), src_max_len) for x in source_seqs])
            self.target_lens = torch.LongTensor([min(len(x), tgt_max_len) for x in target_seqs])
            # self.lengths = torch.LongTensor([min(len(x), max_poetry_length) - 1 for x in data]) # -1?

            # pad data
            src_pad_len = min(max(self.source_lens), src_max_len)
            tgt_pad_len = min(max(self.target_lens), tgt_max_len)

            self.source = torch.zeros((self.lens, src_pad_len)).long()
            self.target = torch.zeros((self.lens, tgt_pad_len)).long()
            for i in range(self.lens):
                TL = min(self.source_lens[i], src_max_len)
                self.source[i, :TL] = torch.LongTensor(source_seqs[i][:TL])

                L = min(self.target_lens[i], tgt_max_len)
                self.target[i, :L] = torch.LongTensor(target_seqs[i][:L])  # 0:L
                # self.target[i, :L] = torch.LongTensor(data[i][1:(L + 1)]) # 1:L+1 target?

            self.source = self.source.to(device)
            self.target = self.target.to(device)
            self.source_lens = self.source_lens.to(device)
            self.target_lens = self.target_lens.to(device)

    def __len__(self):
        return self.lens_1

    def __getitem__(self, index):
        if not self.test:
            out = (self.source[index, :], self.source_lens[index], self.target[index, :],
                   self.target_lens[index])
        else:
            out = ((self.source_1[index, :], self.source_lens_1[index]), 
                   (self.source_2[index, :], self.source_lens_2[index]))
        return out
