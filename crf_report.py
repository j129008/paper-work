from lib.learner import Learner
from lib.ensumble_learner import Bagging, Boosting
from lib.feature import *
from copy import deepcopy
import csv

path = './data/data_shuffle.txt'
result_table = csv.writer( open('./csv/crf_report.csv', 'w') )

base_data = Context(path, k=5)

# baseline
man = Learner(base_data)
man.train()
man.report()
score = man.get_score()
result_table.writerow(['baseline', score['P'], score['R'], score['f1']])

# data size
result_table.writerow(['data size'])
for train_size in [0.1*i for i in range(1, 11)]:
    man = Learner(base_data)
    man.train(sub_train=train_size)
    print('data size:', train_size)
    man.report()
    score = man.get_score()
    result_table.writerow([train_size, score['P'], score['R'], score['f1']])

# context
result_table.writerow(['context', 'k'])
for n in [1, 2, 3]:
    for k in range(1, 10):
        context_data = Context(path, k=k, n_gram=n)
        man = Learner(context_data)
        man.train()
        print('context', n, k)
        man.report()
        score = man.get_score()
        result_table.writerow([n, k, score['P'], score['R'], score['f1']])

# tdiff
tdiff = Tdiff(path)
man = Learner(tdiff + base_data)
man.train()
print('tdiff')
man.report()
score = man.get_score()
result_table.writerow(['tdiff', score['P'], score['R'], score['f1']])

# pmi
pmi = MutualInfo(path)
man = Learner(pmi + base_data)
man.train()
print('pmi')
man.report()
score = man.get_score()
result_table.writerow(['pmi', score['P'], score['R'], score['f1']])

# tdiff + pmi
man = Learner(pmi + tdiff + base_data)
man.train()
print('tdiff+pmi')
man.report()
score = man.get_score()
result_table.writerow(['tdiff + mi', score['P'], score['R'], score['f1']])

# rhyme
result_table.writerow(['rhyme'])
type_list = ['反切', '聲母', '韻目', '調', '等', '呼', '韻母']
rhyme = Rhyme(path, './data/rhyme.txt', './pickles/rhyme_list.pkl', type_list)

for t in type_list:
    t_data = rhyme.get_feature(t)
    man = Learner(t_data + base_data)
    man.train()
    print('rhyme:', t)
    man.report()
    score = man.get_score()
    result_table.writerow([t, score['P'], score['R'], score['f1']])

# name tag
office = Label(path, lab_name='office', lab_file='./ref/tang_name/tangOffice.clliu.txt')
nianhao = Label(path, lab_name='nianhao', lab_file='./ref/tang_name/tangReignperiods.clliu.txt')
address = Label(path, lab_name='address', lab_file='./ref/tang_name/tangAddresses.clliu.txt')
compare_list = [ (office, 'office'), (nianhao, 'nianhao'), (address, 'address') ]

for data, data_name in compare_list:
    man = Learner(data + base_data)
    man.train()
    print('tag name:', data_name)
    man.report()
    score = man.get_score()
    result_table.writerow([data_name, score['P'], score['R'], score['f1']])

# seg size
for k in range(1, 7):
    base_data = Context(path, k=k)
    base_data.segment(length=1)
    base_data.shuffle()
    man = Learner(base_data)
    man.train()
    print('k:', k)
    man.report()
    score = man.get_score()
    result_table.writerow(['baseline', k, score['P'], score['R'], score['f1']])

    man = Bagging(base_data)
    man.train()
    print('bagging:', k)
    man.report()
    score = man.get_score()
    result_table.writerow(['bagging', k, score['P'], score['R'], score['f1']])

    man = Boosting(base_data)
    man.train()
    print(man.alpha_list)
    print('boost:', k)
    man.report()
    score = man.get_score()
    result_table.writerow(['boosting', k, score['P'], score['R'], score['f1']])
