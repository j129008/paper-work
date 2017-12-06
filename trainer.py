from lib.crf import CRF
from lib.data import Data
from lib.learner import Learner
from lib.feature import Feature
from lib.ensumble_learner import Boosting
from lib.ensumble_learner import Bagging
from tqdm import tqdm as bar
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


p_list = []
r_list = []
f1_list = []
x_list = []

for x in bar( range(2, 5) ):
    man = Boosting('./data/data4.txt')
    man.load_feature(funcs=[Feature.context], params=[{'k':x, 'n_gram':2}])
    man.train()
    p_list.append(man.get_score()['P'])
    r_list.append(man.get_score()['R'])
    f1_list.append(man.get_score()['f1'])
    x_list.append(x)

plt.subplot(311).title.set_text('precision')
plt.plot(x_list, p_list)

plt.subplot(312).title.set_text('recall')
plt.plot(x_list, r_list)

plt.subplot(313).title.set_text('f1')
plt.plot(x_list, f1_list)

plt.tight_layout()
plt.savefig('/mnt/d/trainer.png',dpi=300,format="png")
