from sklearn.model_selection import train_test_split

text_list = [ line for line in open('./data/data_shuffle.txt', 'r') ]
train_list, test_list = train_test_split(text_list, test_size=0.3, shuffle=False)

fw = open('./data/train.txt', 'w')
fw.writelines(train_list)
fw.close()

fw = open('./data/test.txt', 'w')
fw.writelines(test_list)
fw.close()
