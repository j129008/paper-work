from lib.learner import Learner
from lib.ensumble_learner import Bagging, Boosting
from lib.feature import *
from copy import deepcopy
import csv

path = './data/data_lite.txt'
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
    for k in bar(range(1, 10)):
        context_data = Context(path, k=k, n_gram=n)
        man = Learner(context_data)
        man.train()
        score = man.get_score()
        result_table.writerow([n, k, score['P'], score['R'], score['f1']])

# tdiff
tdiff = Tdiff(path)
man = Learner(tdiff + base_data)
man.train()
man.report()
score = man.get_score()
result_table.writerow(['tdiff', score['P'], score['R'], score['f1']])

# pmi
pmi = MutualInfo(path)
man = Learner(pmi + base_data)
man.train()
man.report()
score = man.get_score()
result_table.writerow(['pmi', score['P'], score['R'], score['f1']])

# tdiff + pmi
man = Learner(pmi + tdiff + base_data)
man.train()
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
    man.report()
    score = man.get_score()
    result_table.writerow([t, score['P'], score['R'], score['f1']])

# seg size
result_table.writerow(['seg size'])
for seg_size in [1, 10, 100]:
    seg_data = deepcopy(base_data)
    seg_data.segment(length=seg_size)
    man = Learner(base_data)
    man.train()
    man.report()
    score = man.get_score()
    result_table.writerow([seg_size, score['P'], score['R'], score['f1']])

# ensemble
man = Bagging(base_data)
man.train()
man.report()
score = man.get_score()
result_table.writerow(['bagging', score['P'], score['R'], score['f1']])

_context = deepcopy(base_data)
_context.segment(length=10)
man = Boosting(_context)
man.train()
print(man.alpha_list)
man.report()
score = man.get_score()
result_table.writerow(['boosting', score['P'], score['R'], score['f1']])
