from lib.ensumble_learner import Bagging
from lib.arg import crf_arg, crf_data

parser = crf_arg()
args = parser.parse_args()
data = crf_data(args)
data.segment(length=1)

man = Bagging(data)
man.train()
man.report()
