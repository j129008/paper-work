from lib.learner import Learner
from lib.arg import crf_arg, crf_data

parser = crf_arg()
parser.add_argument('-iter', dest='iter', type=int, default=None, help='CV tune per iteration')
parser.add_argument('-cv', dest='cv', type=int, default=None, help='cross validation')
parser.add_argument('-c1', dest='c1', type=float, default=0.0, help='The coefficient for L1 regularization')
parser.add_argument('-c2', dest='c2', type=float, default=1.0, help='The coefficient for L2 regularization')
parser.add_argument('-subtrain', dest='subtrain', type=float, default=1.0, help='set training data size')
args = parser.parse_args()
data = crf_data(args)

man = Learner(data)
man.split_data(train_size=args.trainsplit)

if args.cv != None and args.iter != None:
    man.train_CV(cv=args.cv, n_iter=args.iter)
else:
    man.train(c1=args.c1, c2=args.c2, sub_train=args.subtrain)

if args.trainsplit < 1.0:
    man.report()

if args.save != None:
    man.save(args.save)
