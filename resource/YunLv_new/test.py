line = '一,yi,y,i,4E00,1\n'
dicts = []
words = []
cnt = 0
if 1:
    flag = 0
    line = line.strip('\n')
    line = line.split(',')
    word = line[0]
    sheng = line[2]
    yun = line[3]
    for dict in dicts:
        if word == dict['word']:
            dict['sheng'] = sheng
            dict['yun'] = yun
            flag = 1
    if flag == 0:
        dicts.append({'word': word, 'pz': '', 'sheng': sheng, 'yun': yun})
        words.append(word)
        cnt += 1
        print(cnt)
print(dicts)