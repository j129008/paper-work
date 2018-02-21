from lib.crf import CRF
from lib.data import Data
from lib.learner import Learner
from lib.feature import Context
from lib.ensumble_learner import Boosting
from lib.ensumble_learner import Bagging
from pprint import pprint
import csv

fp = csv.writer( open('precision.csv', 'w') )
fr = csv.writer( open('recall.csv', 'w') )
data = Context('./data/data4.txt')
man = Bagging(data, random_state=0)

P_list = []
R_list = []
for size in [0.1, 0.2, 0.3, 0.4]:
    P_voter_list = []
    R_voter_list = []
    for voter in range(3, 7):
        print( 'voter: ', voter, 'size: ', size )
        man.train(voter=voter, train_size=size)
        pprint(man.get_score())
        R_voter_list.append(man.get_score()['R'])
        P_voter_list.append(man.get_score()['P'])
    P_list.append(P_voter_list)
    R_list.append(R_voter_list)
fp.writerows(P_list)
fr.writerows(R_list)
man.baseline()
