from lib.feature import *

def crf_arg():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-pmi', action='store_true')
    parser.add_argument('-tdiff', action='store_true')
    parser.add_argument('-rhy', dest='rhy', default=None, help='反切,聲母,韻目,調,等,呼,韻母')
    parser.add_argument('-list', dest='list', default=None)
    parser.add_argument('-k', dest='k', default=1, type=int)
    parser.add_argument('-ngram', dest='ngram', default=2, type=int)
    parser.add_argument('-i', dest='input')
    return parser

def crf_data(args):
    path = args.input
    data = Context(path, k=args.k, n_gram=args.ngram)
    if args.pmi:
        data += MutualInfo(path)
    if args.tdiff:
        data += Tdiff(path)
    if args.rhy != None:
        data += Rhyme(path, '../data/rhyme.txt', '../pickles/rhyme_list.pkl', args.rhy.split(','))
    if args.list != None:
        for name_path in args.list.split(','):
            lab_name, lab_file = name_path.split(':')
            data += Label(path, lab_name=lab_name, lab_file=lab_file)
    return data
