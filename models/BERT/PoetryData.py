# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SRC_MAX_LENGTH = 32
TGT_MAX_LENGTH = 41

def paired_collate_fn(insts):
    return default_collate(insts)

def collate_fn(insts):
    return default_collate(insts)

class PoetryData(Dataset):
    def __init__(self, data, src_max_len=SRC_MAX_LENGTH, tgt_max_len=TGT_MAX_LENGTH, test=False):
        
        self.test = test
        self.lens = len(data)

        source_seqs = [i[0] for i in data]
        target_seqs = [i[1] for i in data]
        
        self.source = torch.Tensor(source_seqs)
        
        # pad data
        self.target_lens = torch.LongTensor([min(len(x), tgt_max_len) for x in target_seqs])
        tgt_pad_len = min(max(self.target_lens), tgt_max_len)
        self.target = torch.zeros((self.lens, tgt_pad_len)).long()
        for i in range(self.lens):
            L = min(self.target_lens[i], tgt_max_len)
            self.target[i, :L] = torch.LongTensor(target_seqs[i][:L])  # 0:L
            # self.target[i, :L] = torch.LongTensor(data[i][1:(L + 1)]) # 1:L+1 target?

        self.source = self.source.to(device)
        self.target = self.target.to(device)

    def __len__(self):
        return self.lens

    def __getitem__(self, index):
        if not self.test:
            out = (self.source[index], self.target[index])
        else:
            out = (self.source[index, :], self.source_lens[index])
        return out
