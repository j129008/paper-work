from lib.learner import Learner
from lib.arg import crf_arg, crf_data
from lib.metric import pred2text
import pickle

parser = crf_arg()
args = parser.parse_args()
data = crf_data(args)
union = lambda data: [ ins for chap in data for ins in chap]
model = pickle.load(open(args.save, 'rb'))
print(pred2text(args.input, union(model.predict(data.X))))
