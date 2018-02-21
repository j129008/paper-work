from lib.learner import Learner
from lib.ensumble_learner import Bagging, Boosting
from lib.feature import *
from tqdm import tqdm as bar
import csv
import sys

learn_method = sys.argv[1]
result_table = csv.writer( open('/mnt/d/progress/'+learn_method+'.csv', 'w') )
result_table.writerow(['context', 'precision', 'recall', 'f1'])
path = './data/data2.txt'
print('start', learn_method)

def experiment(data, tune=False, random_state=0):
    if learn_method == 'Bagging':
        man = Bagging(data, random_state=random_state)
        tune = False
    elif learn_method == 'Boosting':
        man = Boosting(data, random_state=random_state)
        tune = False
    else:
        man = Learner(data, random_state=random_state)
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
context_rhyme = [['context + rhyme']]
context_office = [['context + office label']]

mi_data = MutualInfo(path)
tdiff_data = Tdiff(path)
rhyme_data = Rhyme(path, './data/rhyme.txt', './pickles/rhyme_list.pkl', ['反切', '聲母', '韻目', '調', '等', '呼', '韻母'])
office_data = Label(path, 'office', './ref/known/office2.txt')

for k in bar(range(1, 6)):
    print('context')
    context_data = Context(path, k=k)
    score = experiment(context_data)
    context.append([k, score['P'], score['R'], score['f1']])

    print('context + MI')
    score = experiment(context_data + mi_data)
    context_mi.append([k, score['P'], score['R'], score['f1']])

    print('context + t-diff')
    score = experiment(context_data + tdiff_data)
    context_tdiff.append([k, score['P'], score['R'], score['f1']])

    print('context + mutual infomation + t-diff')
    score = experiment(context_data + tdiff_data + mi_data)
    context_mi_tdiff.append([k, score['P'], score['R'], score['f1']])

    print('context + rhyme')
    score = experiment(context_data + rhyme_data)
    context_rhyme.append([k, score['P'], score['R'], score['f1']])

    print('context + word label')
    score = experiment(context_data + office_data)
    context_office.append([k, score['P'], score['R'], score['f1']])

result_table.writerows(context)
result_table.writerows(context_mi)
result_table.writerows(context_tdiff)
result_table.writerows(context_mi_tdiff)
result_table.writerows(context_rhyme)
result_table.writerows(context_office)
