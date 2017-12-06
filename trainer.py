from lib.crf import CRF
from lib.data import Data
from lib.learner import Learner
from lib.feature import Feature
from lib.ensumble_learner import Boosting
from lib.ensumble_learner import Bagging
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

man = Bagging('./data/data4.txt')
man.load_feature(funcs=[Feature.context], params=[{'k':2, 'n_gram':2}])

p_list = []
r_list = []
v_list = []

for voter in range(2, 10):
    man.train(voter=voter)
    p_list.append(man.get_score()['P'])
    r_list.append(man.get_score()['R'])
    v_list.append(voter)

plt.subplot(211)
plt.plot(v_list, p_list)

plt.subplot(212)
plt.plot(v_list, r_list)

plt.savefig('/mnt/d/trainer.png',dpi=300,format="png")
