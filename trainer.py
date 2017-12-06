from lib.crf import CRF
from lib.data import Data
from lib.learner import Learner
from lib.feature import Feature
from lib.ensumble_learner import Boosting

boosting = Boosting('./data/data4.txt')
boosting.feature_loader(funcs=[Feature.context], params=[[]])
boosting.train()
boosting.report()
