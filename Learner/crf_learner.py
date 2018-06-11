from lib.learner import Learner
from lib.arg import crf_arg, crf_data

parser = crf_arg()
parser.add_argument('-iter', dest='iter', type=int, default=None)
parser.add_argument('-cv', dest='cv', type=int, default=None)
parser.add_argument('-c1', dest='c1', type=float, default=0.0)
parser.add_argument('-c2', dest='c2', type=float, default=1.0)
parser.add_argument('-size', dest='size', type=float, default=1.0)
args = parser.parse_args()
data = crf_data(args)

man = Learner(data)

if args.cv != None and args.iter != None:
    man.train_CV(cv=args.cv, n_iter=args.iter)
    man.report()
else:
    man.train(c1=args.c1, c2=args.c2, sub_train=args.size)
    man.report()
