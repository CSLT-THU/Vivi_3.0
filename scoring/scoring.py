import json
import os
import random


names = [
    'base',
    'base+rule',
    'soft',
    'soft+rule',
    'jiuge'
    # 'gt'
]

# for combine_result15
# files = [
#     '../result/07261942_17_Seq2seq_12_ep=6_loss=144.80_013103.txt',
# 
#     '../result/07261942_17_Seq2seq_12_ep=6_loss=144.80_110340.txt',
# 
#     # '../result/07270727_17_Seq2seq_12_ep=6_loss=150.58_023446.txt', # w1=3
#     '../result/07270727_17_Seq2seq_12_ep=6_loss=150.58_023620.txt', # w1=6
# 
#     '../result/07270727_17_Seq2seq_12_ep=6_loss=150.58_110534.txt',
# 
#     '../result/jiuge_result2.txt'
#     # '../result/test_10k_1k_gt.txt'
# ]

# for combine_result15
# num_li_all = [3,22,24,35,41,43,48,59,67,69,72,76,80,88,92,95,96,98,104,122,
#               125,128,130,143,145,155,156,163,168,174,177,197,203,205,210,211,221,227,253,256,
#               284,286,289,303,309,316,319,325,326,337,342,355,362,365,385,394,406,423,428,429,
#               430,431,432,435,437,447,463,472,481,514,534,540,543,557,563,567,577,585,612,709]



# files = [
#     '../result/07261942_17_Seq2seq_12_ep=6_loss=144.80_013103.txt',
#     '../result/07261942_17_Seq2seq_12_ep=6_loss=144.80_110340.txt',
#     '../result/07270727_17_Seq2seq_12_ep=6_loss=150.58_023446.txt',
#     '../result/07270727_17_Seq2seq_12_ep=6_loss=150.58_110534.txt',
#     '../result/test_10k_1k_gt.txt'
# ]
# 
# num_li = [1, 3, 6, 7, 12, 15, 22, 23, 24, 35, 39, 41, 45, 48, 59, 62, 65, 72, 75, 76, 80, 
#           87, 88, 89, 92, 94, 96, 98, 99, 102, 104, 106, 107, 110, 115, 117, 122,124, 125, 128, 131,
#           141, 143, 144, 145, 147, 148, 149, 152, 155, 156, 157, 158, 163, 172, 174, 176, 177, 179, 
#           181, 183, 190, 194, 198, 199 ]


# for combine_result23
files = [
    '../result/0802164334_Seq2seq_12_ep=3_loss=150.38_140507.txt',
    '../result/0802164334_Seq2seq_12_ep=3_loss=150.38_134908.txt', 
    '../result/0802164555_Seq2seq_12_ep=3_loss=154.49_142759.txt', # w1=4
    '../result/0802164555_Seq2seq_12_ep=3_loss=154.49_143401.txt',  # w1=4
    '../result/jiuge_result2.txt'
]

num_li_all = [
    7,15,35,45,48,51,67,69,80,88,
    96,103,104,124,125,133,134,143,145,148,
    156,158,163,170,174,176,194,195,210,221,
    226,237,240,243,268,284,285,289,290,294,
    301,305,319,333,338,347,381,389,410,429,
    431,432,440,481,487,490,514,527,529,543,
    546,550,557,574,577,585,588,595,605,631,
    646,671,681,703,713,737,748,751,773,792
]

root = '1031k_0831_3/'
n_questionair = 20
n_question = 4

for k in range(n_questionair):
    dir = root + str(k+1) + '/'
    save = dir + 'scoring.txt'
    save_rand = dir + 'scoring_rand.txt'
    save_idx = dir + 'idx.json'

    name_dict = {}
    for i in range(len(files)):
        name_dict[files[i]] = names[i]

    li = [[] for i in range(n_question)]

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(save, 'w', encoding='utf-8') as f1:
        for file in files:
            f1.write('file: ' + file)
            f1.write('\tsettings: ' + name_dict[file])
            f1.write('\n')
        f1.write('\n')

    # examples
    num_li = num_li_all[ (k*n_question) : (k*n_question+n_question) ]
    for file in files:
        # file = files[i]
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for l in range(len(lines)):
                if l in num_li:  #
                    idx = num_li.index(l)
                    li[idx].append(lines[l - 1])

    li_shuffle = [[] for i in range(len(num_li))]
    idxs_li = []
    for i in range(len(li)):
        idxs = [1,2,3,4,5]
        random.shuffle(idxs)
        for j in idxs:
            li_shuffle[i].append(li[i][j - 1])
        idxs_li.append(idxs)

    with open(save, 'a', encoding='utf-8') as f0:
        n = 0
        for poems in li:
            n += 1
            f0.write(str(n) + '-' + str(num_li[n - 1]) + '\n')
            for i in range(len(poems)):
                poem = poems[i]
                s1 = names[i]
                # s1 = str(files[i]).split('.txt')[0]
                s2 = poem
                f0.write('%-25s%-20s' % (s1, s2))
        print('saved at: ' + save)

    with open(save_rand, 'w', encoding='utf-8') as f1:
        n = 0
        for poems in li_shuffle:
            n += 1
            f1.write(str(n) + '\n')
            for poem in poems:
                s = poem.split(' ==== ')[1].replace('\n', '').split('/')
                f1.write(s[0] + '，' +  s[1] + '。' + s[2] + '，' + s[3] + '。' + '\n')

    with open(save_idx, 'w') as f2:
        json.dump(idxs_li, f2)
    