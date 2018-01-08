from lib.feature import Feature
from lib.ensumble_learner import Boosting

MAN = Boosting('./data/data4.txt')
MAN.load_feature(funcs=[Feature.context, Feature.t_diff, Feature.mi_info, Feature.rhyme], params=[{'k':1, 'n_gram':2}, {}, {}, {'path':'./data/rhyme.txt', 'pkl_path':'./pickles/rhyme_list.pkl', 'rhy_type_list':['調']}])
MAN.train()
print(MAN.alpha_list)
MAN.report()
MAN.baseline()
