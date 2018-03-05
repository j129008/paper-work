from lib.learner import Learner
from lib.ensumble_learner import Bagging, Boosting
from lib.feature import *
from tqdm import tqdm as bar
from copy import deepcopy
import csv
import sys

for n in range(1,4):
    result_table = csv.writer( open('/mnt/d/progress/data_need_'+str(n)+'_gram.csv', 'w') )
    path = './data/data_proc.txt'
    result_table.writerow(['k', 'precision', 'recall', 'f1'])
    print('ngram:', n)

    for k in bar(range(1, 10)):
        context_data = Context(path, k=k, n_gram=n)
        context_data.shuffle()
        man = Learner(context_data)
        man.train(sub_train=0.5)
        score = man.get_score()
        result_table.writerow([k, score['P'], score['R'], score['f1']])
