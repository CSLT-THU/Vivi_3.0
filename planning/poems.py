# -*- coding:utf-8 -*-

import os
from random import shuffle

from planning.char_dict import CharDict
from planning.data_utils import split_sentences, split_1031k
from planning.paths import raw_dir, poems_path, check_uptodate

_corpus_list = ['poem_1031k.txt']
# _corpus_list = ['qts_tab.txt']
# _corpus_list = ['qts_tab.txt', 'qss_tab.txt', 'qsc_tab.txt', 'qtais_tab.txt',
#                 'yuan.all', 'ming.all', 'qing.all', 'poem_1031k.txt']


# 诗集poem的语料库，可多选


def _gen_poems():
    print("Parsing poems ...")
    char_dict = CharDict()
    with open(poems_path, 'w', encoding='UTF-8') as fout:
        for corpus in _corpus_list:
            with open(os.path.join(raw_dir, corpus), 'r', encoding='UTF-8') as fin:
                for line in fin.readlines(): # for poem_1031k 
                # for line in fin.readlines()[1:]:

                    if corpus == 'poem_1031k.txt': # 格式不同
                        sentences = split_1031k(line)
                    else:
                        sentences = split_sentences(line.strip().split()[-1])
                    
                    all_char_in_dict = True
                    for sentence in sentences:
                        for ch in sentence:
                            # print(ch) #
                            if char_dict.char2int(ch) < 0:
                                all_char_in_dict = False
                                break
                        if not all_char_in_dict:
                            break
                    if all_char_in_dict:
                        fout.write(' '.join(sentences) + '\n')                
                
            print("Finished parsing %s." % corpus)


class Poems:

    def __init__(self):
        if not check_uptodate(poems_path):
            _gen_poems()
        self.poems = []
        with open(poems_path, 'r', encoding='UTF-8') as fin:
            for line in fin.readlines():
                self.poems.append(line.strip().split())

    def __getitem__(self, index):
        if index < 0 or index >= len(self.poems):
            return None
        return self.poems[index]

    def __len__(self):
        return len(self.poems)

    def __iter__(self):
        return iter(self.poems)

    def shuffle(self):
        shuffle(self.poems)


# For testing purpose.
if __name__ == '__main__':
    poems = Poems()
    for i in range(10):
        print(' '.join(poems[i]))
