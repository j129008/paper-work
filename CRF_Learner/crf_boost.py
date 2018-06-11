from lib.ensumble_learner import Boosting
from lib.arg import crf_arg, crf_data

parser = crf_arg()
parser.add_argument('-trainsplit', dest='trainsplit', type=float, default=0.7)
parser.add_argument('-savemodel', dest='save', default=None)
parser.add_argument('-seg', dest='seg', type=int, default=1)
args = parser.parse_args()
data = crf_data(args)
data.segment(length=args.seg)

man = Boosting(data)
man.split_data(train_size=args.trainsplit)
man.train()
if args.save != None:
    man.save(args.save)
print(man.alpha_list)

if args.trainsplit < 1:
    man.report()
