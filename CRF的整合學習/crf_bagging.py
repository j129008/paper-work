from lib.ensumble_learner import Bagging
from lib.arg import crf_arg, crf_data

parser = crf_arg()
parser.add_argument('-seg', dest='seg', type=int, default=1)
parser.add_argument('-voter', dest='voter', type=int, default=8)
parser.add_argument('-votersize', dest='votersize', type=float, default=0.4)
args = parser.parse_args()
data = crf_data(args)
data.segment(length=args.seg)

man = Bagging(data)
man.split_data(train_size=args.trainsplit)
man.train(voter=args.voter, train_size=args.votersize)

if args.trainsplit < 1:
    man.report()
