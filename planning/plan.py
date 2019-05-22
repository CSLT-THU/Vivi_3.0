# -*- coding:utf-8 -*-

import os
import random
from random import shuffle
import time

import jieba
from gensim import models

from planning.char_dict import end_of_sentence, start_of_sentence
from planning.data_utils import split_sentences, NUM_OF_SENTENCES
from planning.paths import plan_history_path, plan_data_path, check_uptodate
from planning.paths import save_dir
from planning.poems import Poems
from planning.rank_words import RankedWords
from planning.segment import Segmenter

_ancient_model_path = os.path.join(save_dir, 'ancient_model_5.bin')
_modern_model_path = os.path.join(save_dir, 'sgns.baidubaike.bigram-char')
# _modern_model_path = os.path.join(save_dir, 'sgns.literature.bigram-char')
# modern word2vec model download: https://github.com/Embedding/Chinese-Word-Vectors


def gen_train_data():
    print("Generating training data ...")
    segmenter = Segmenter()
    poems = Poems()
    poems.shuffle()
    ranked_words = RankedWords()
    plan_data = []  # 从poem.txt中把筛选出来的诗做关键词提取，按wordrank分优先级
    plan_history = []  # 记录每句诗以及从诗中提取的planning word，做参照查看
    for poem in poems:
        if len(poem) != 4:
            continue  # Only consider quatrains.
        valid = True
        context = start_of_sentence()
        gen_lines = []
        keywords = []
        for sentence in poem:
            if len(sentence) != 7: 
                print('len sen != 7') ####
                valid = False
                break
            words = list(filter(lambda seg: seg in ranked_words,
                                segmenter.segment(sentence)))
            if len(words) == 0:
                print('len word = 0') ####
                valid = False
                break
            keyword = words[0]
            for word in words[1:]:
                if ranked_words.get_rank(word) < ranked_words.get_rank(keyword):
                    keyword = word
            gen_line = sentence + end_of_sentence() + \
                       '\t' + keyword + '\t' + context + '\n'
            gen_lines.append(gen_line)
            keywords.append(keyword)
            context += sentence + end_of_sentence()
        
        print(valid) ####
        if valid:
            plan_data.append('\t'.join(keywords) + '\n')
            plan_history.extend(gen_lines)
    with open(plan_data_path, 'w', encoding='UTF-8') as fout:
        for line in plan_data:
            fout.write(line)
    with open(plan_history_path, 'w', encoding='UTF-8') as fout:
        for line in plan_history:
            fout.write(line)


def train_planner():
    print("Training Word2Vec-based planner ...")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not check_uptodate(plan_data_path):
        gen_train_data()
    word_lists = []
    with open(plan_data_path, 'r', encoding='UTF-8') as fin:
        for line in fin.readlines():
            word_lists.append(line.strip().split('\t'))
    model = models.Word2Vec(word_lists, size=512, min_count=5)
    model.save(_ancient_model_path)


class Planner:

    def __init__(self):
        self.ranked_words = RankedWords()
        if not os.path.exists(_ancient_model_path):
            raise Exception("ancient model doesn't exist")
            # train_planner()
        if not os.path.exists(_modern_model_path):
            raise Exception("Please download Chinese Word vector in save dir")

        self.ancient_model = models.Word2Vec.load(_ancient_model_path)
        self.modern_model = models.KeyedVectors.load_word2vec_format(_modern_model_path)

    def plan(self, text):
        return self._expand(self._extract(text))

    def _extract(self, text):

        def extract_from_sentence(initial_words):
            print('input:', initial_words)
            oov = []
            for word in initial_words:
                if not self.modern_model.__contains__(word):
                    oov.append(word)
            for word in oov:
                initial_words.update(word)  # 将word拆分后加入
                initial_words.remove(word)
            print('modify:', initial_words)
            
            similars = self.modern_model.most_similar(positive=initial_words, topn=10)
            similars = sorted(similars, key=lambda x: x[1] * random.random())
            for similar in similars:
                initial_words.add(similar[0])
            return filter(lambda w: w in self.ranked_words,
                          initial_words)
        
        # def extract_from_sentence(sentence):
        #     initial_words = set(jieba.lcut(sentence))
        #     # similars = self.ancient_model.most_similar(positive=initial_words, topn=10)
        #     similars = self.modern_model.most_similar(positive=initial_words, topn=10)
        #     similars = sorted(similars, key=lambda x: x[1] * random.random())
        #     for similar in similars:
        #         initial_words.add(similar[0])
        #     return filter(lambda w: w in self.ranked_words,
        #                   initial_words)

        keywords = set() 
        initial_words = set(text.split('-')) #
        keywords.update(extract_from_sentence(initial_words)) #
        # for sentence in split_sentences(text):
        #     keywords.update(extract_from_sentence(sentence))
        if len(keywords) > NUM_OF_SENTENCES:
            keywords = set(random.sample(keywords, NUM_OF_SENTENCES))

        return keywords

    def _expand(self, keywords):
        if len(keywords) < NUM_OF_SENTENCES:
            filtered_keywords = list(filter(lambda w: w in \
                                                      self.ancient_model.wv, keywords))
            if len(filtered_keywords) > 0:
                similars = self.ancient_model.wv.most_similar(
                    positive=filtered_keywords)
                # Sort similar words in decreasing similarity with randomness.
                similars = sorted(similars, key=lambda x: x[1] * random.random())
                for similar in similars:
                    keywords.add(similar[0])
                    if len(keywords) == NUM_OF_SENTENCES:
                        break
            prob_sum = sum(1. / (i + 1) \
                           for i, word in enumerate(self.ranked_words) \
                           if word not in keywords)
            rand_val = prob_sum * random.random()
            word_idx = 0
            s = 0
            while len(keywords) < NUM_OF_SENTENCES \
                    and word_idx < len(self.ranked_words):
                word = self.ranked_words[word_idx]
                s += 1.0 / (word_idx + 1)
                if word not in keywords and rand_val < s:
                    keywords.add(word)
                word_idx += 1
        results = list(keywords)
        shuffle(results)
        return results

print('start loading planning model')
start = time.time()
planner = Planner()
print('load planning model done. time:', time.time()-start)
    
# For testing purpose.
if __name__ == '__main__':
    print('start loading model')
    start = time.time()
    planner = Planner()
    print('load done. time:', time.time()-start)
    inputs = [
              # "春天到了，桃花开了。",
              # "举杯饮酒，思乡情怯",
              # "牧童遥指杏花村",
              # "中秋节的夜晚，想起故乡的月亮",
              # "奥巴马会见科比，一同打篮球，踢毽子",
              # "纵情山河万里，肆意九州五岳",
              # "燕子",
              "燕子 春泥 杨柳",
              "燕子春泥杨柳",
              "燕子，春泥，杨柳",
              "燕子 鸟语花香 风和日丽 春暖花开",
              "燕子鸟语花香风和日丽春暖花开",
              "燕子，鸟语花香，风和日丽，春暖花开",
              "秋风 夕阳 马 古道",
              "秋风夕阳马古道",
              "秋风，夕阳，马，古道",
              "秋风 夕阳",
              "秋风夕阳",
              "秋风,夕阳"
              ]

    for input in inputs:
        keywords = planner.plan(input)
        print(keywords)
        # keywords = planner._extract(input)
        # print(keywords)
        # print('extract:')
        # print('{0:{2}^12}  ->  {1}\t'.format(input, keywords, chr(12288)))
        # 
        # keywords = planner._expand(keywords)
        # print(keywords)
        # print('expand:')
        # print('{0:{2}^12}  ->  {1}\t'.format(input, keywords, chr(12288)))
        # print('\n', '------------------------', '\n')
