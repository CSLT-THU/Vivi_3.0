with open('pinyin_oov.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        li = line.split(' ')
print(li)
shengmu = ['b', 'p', 'm' ,'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
'j', 'q', 'x', 'z', 'c', 's', 'r', 'y', 'w']
with open('hzpy.txt', 'a', encoding='utf-8') as f2:
# with open('pinyin_oov2.txt', 'w', encoding='utf-8') as f2:
    for item in li:
        if len(item) > 1:
            if item[2:].startswith('h'): # zh ch sh
                print(item)
                f2.write(item[0] + ',' + item[1:] + ',' + item[1:3] + ',' + item[3:])
                f2.write('\n')
            elif item[1] in shengmu: # 有声母（不是“饿”之类的）
                f2.write(item[0] + ',' + item[1:] + ',' + item[1] + ',' + item[2:])
                f2.write('\n')
            else:
                print(item)
                f2.write(item[0] + ',' + item[1:] + ',0,' + item[1:]) # 声母用0代替
                f2.write('\n')
        else:
            print(item)

