from lib.learner import Learner
from lib.feature import *
from lib.data import PklData
import csv

result_table = csv.writer( open('result_table.csv', 'w') )
result_table.writerow(['context', 'precision', 'recall', 'f1'])
path = './data/data4.txt'

def experiment(data, tune=True):
    man = Learner(data, random_state=0)
    if tune == True:
        man.train_CV()
    else:
        man.train()
    man.report()
    return man.get_score()

context = [['context']]
context_mi = [['context + MI']]
context_tdiff = [['context + tdiff']]
context_mi_tdiff = [['context + tdiff + mi']]
for k in range(1, 6):
    # context
    context_data = Context(path, k=k)
    score = experiment(context_data)
    context.append([k, score['P'], score['R'], score['f1']])
    # context + MI
    mi_data = MutualInfo(path)
    score = experiment(context_data + mi_data)
    context_mi.append([k, score['P'], score['R'], score['f1']])
    # context + t-diff
    tdiff_data = Tdiff(path)
    score = experiment(context_data + tdiff_data)
    context_tdiff.append([k, score['P'], score['R'], score['f1']])
    # context + mutual infomation + t-diff
    score = experiment(context_data + tdiff_data + mi_data)
    context_mi_tdiff.append([k, score['P'], score['R'], score['f1']])


result_table.writerows(context)
result_table.writerows(context_mi)
result_table.writerows(context_tdiff)
result_table.writerows(context_mi_tdiff)
# rhyme

# tag

# ensumble
# bagging
# boosting

# svm
# random forest
# deep learning
