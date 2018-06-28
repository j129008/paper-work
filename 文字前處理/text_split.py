from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

parser = ArgumentParser()
parser.add_argument('-i', dest='input', help='input file')
parser.add_argument('-train', dest='train', help='train file')
parser.add_argument('-test', dest='test', help='test file')
parser.add_argument('-ts', dest='trainsplit', default=0.7, type=float, help='train test split size')
args = parser.parse_args()

lines = [ line for line in open(args.input, 'r') ]
train, test = train_test_split(lines, test_size=1.0-args.trainsplit)

f = open(args.train, 'w')
f.writelines(train)
f.close()

f = open(args.test, 'w')
f.writelines(test)
f.close()
