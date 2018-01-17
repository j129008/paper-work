from lib.feature import Feature
from lib.ensumble_learner import Boosting
from lib.learner import Learner
from datetime import datetime

start = datetime.now()

path = './data/data4.txt'
man = Boosting(path, random_state=0)
man.load_feature(funcs=[Feature.context], params=[{'k':1, 'n_gram':2}])
man.train()
print(man.alpha_list)
man.report()

print('two level boost')
man = Learner(path, random_state=0, shuffle=True)
man.load_feature(funcs=[Feature.context], params=[{'k':1, 'n_gram':2}])
man.train()
pred = man.predict(man.X)
err_list = []
for ele in zip(man.Y, pred):
    if ele[0] == ele[1]:
        err_list.append(0)
    else:
        err_list.append(1)

man.load_feature(funcs=[Feature.ext_lab], params=[{'lab_name':'err', 'lab':err_list}])
man.split_data(random_state=0, shuffle=True)
man.train()
man.report()

print('baseline')
man = Learner(path, random_state=0, shuffle=True)
man.load_feature(funcs=[Feature.context], params=[{'k':1, 'n_gram':2}])
man.train()
man.report()

end = datetime.now()
delta = end - start
print( delta.total_seconds() )
