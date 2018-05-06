from random import shuffle

text_list = [ line for line in open('./data/data_proc.txt', 'r') ]
shuffle(text_list)

fw = open('./data/data_shuffle.txt', 'w')
fw.writelines(text_list)
fw.close()

spliter = int(len(text_list) * 0.7)
fw = open('./data/train.txt', 'w')
fw.writelines(text_list[:spliter])
fw.close()

fw = open('./data/test.txt', 'w')
fw.writelines(text_list[spliter:])
fw.close()
