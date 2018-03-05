from lib.learner import Learner
from lib.feature import Context
from tqdm import tqdm as bar
import csv

result_table = csv.writer( open('/mnt/d/data_need.csv', 'w') )
path = './data/data_proc.txt'
result_table.writerow(['data size', 'precision', 'recall', 'f1'])

for train_size in bar([0.1*i for i in range(1, 11)]):
    P = []
    R = []
    f1 = []
    for _ in range(5):
        data = Context(path)
        data.shuffle()
        man = Learner(data)
        man.train(sub_train=train_size)
        print('train size:', train_size)
        man.report()
        score = man.get_score()
        P.append(score['P'])
        R.append(score['R'])
        f1.append(score['f1'])
    avg = lambda values: sum(values)/len(values)
    result_table.writerow([train_size, avg(P), avg(R), avg(f1)])

