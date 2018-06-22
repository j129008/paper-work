import pdb
import re
from collections import Counter
from sklearn_crfsuite import metrics
from lib.data import Data

class Demo:
    def __init__(self, file_name='demo.txt', learner=None):
        f = open(file_name, 'w')
        truth = learner.Y_private
        pred = learner.predict(learner.X_private)
        w = lambda i: learner.X_private[i[0]][i[1]]['0,0']
        miss_list = []
        err_list = []
        for chap_i in range(len(pred)):
            T = ''
            P = ''
            for i in range(len(pred[chap_i])):
                word = w((chap_i,i))
                P += word
                T += word
                if pred[chap_i][i] != truth[chap_i][i]:
                    if pred[chap_i][i] == 'I':
                        P += '　'
                        T += '，'
                    else:
                        P += '，'
                        T += '　'
                else:
                    if pred[chap_i][i] == 'E':
                        P += '，'
                        T += '，'
            f.write('T: '+T+'\n')
            f.write('P: '+P+'\n')
            miss_list.extend(re.findall(r'(...)　', P))
            err_list.extend(re.findall(r'(...)　', T))
        f.write('miss list:\n' + '\n'.join([ str(ele) for ele in Counter(miss_list).most_common(10) ]) + '\n')
        f.write('err list:\n' + '\n'.join([ str(ele) for ele in Counter(err_list).most_common(10) ]) + '\n')

class ErrorAnalyze:
    def __init__(self, file_name='pred.txt', pred=None, feature_data=None):
        f = open(file_name, 'w')
        union = lambda data: [ y for chap_y in data for y in chap_y ]
        truth = union(feature_data.Y_private)
        pred_u = union(pred)
        X_u = union(feature_data.X_private)
        for i in range(len(X_u)):
            if pred_u[i] != truth[i]:
                f.write('{}, T:{}, P:{}, {}\n'.format(X_u[i]['0,0'], truth[i], pred_u[i], X_u[i]))
            else:
                f.write('{}, {}\n'.format(X_u[i]['0,0'], X_u[i]))
            if truth[i] == 'E':
                f.write('\n')

class ErrorCompare:
    def __init__(self, file_name='compare.txt', pred_list=None, feature_data=None):
        f = open(file_name, 'w')
        union = lambda data: [ y for chap_y in data for y in chap_y ]
        truth = union(feature_data.Y_private)
        X_u = union(feature_data.X_private)
        pred_list_u = [ union(pred) for pred in pred_list ]
        for i in range(len(X_u)):
            pred_i = [truth[i]] + [ pred[i] for pred in pred_list_u]
            if len(set(pred_i)) != 1:
                f.write('{}, {}, {}\n'.format(X_u[i]['0,0'], pred_i[1:], X_u[i]))
            else:
                f.write('{}\n'.format(X_u[i]['0,0']) )
            if truth[i] == 'E':
                f.write('\n')

def pred2text(text_path, pred):
    i = 0
    output = ''
    for line in open(text_path):
        _line = line.replace('，', '').strip()
        for char in _line:
            if pred[i] == 'E':
                output += char + '，'
            else:
                output += char
            i += 1
        output += '\n'
    return output

def text2score(ans_path, pred_path):
    ans_data = Data(ans_path)
    pred_data = Data(pred_path)
    print(metrics.flat_classification_report(
        ans_data.Y, pred_data.Y, labels=('I', 'E'), digits=4
    ))
