import pdb
class Demo:
    def __init__(self, file_name='demo.txt', learner=None):
        f = open(file_name, 'w')
        truth = learner.Y_private
        pred = learner.predict(learner.X_private)
        w = lambda i: learner.X_private[i[0]][i[1]]['0,0']
        for chap_i in range(len(pred)):
            T = ''
            P = ''
            for i in range(len(pred[chap_i])):
                word = w((chap_i,i))
                P += word
                T += word
                if pred[chap_i][i] != truth[chap_i][i]:
                    if pred[chap_i] == 'I':
                        P += '　'
                        T += '，'
                    else:
                        P += '，'
                        T += '　'
                else:
                    if pred[chap_i] == 'I':
                        P += '　'
                        T += '　'
                    else:
                        P += '，'
                        T += '，'
            f.write('T: '+T+'\n')
            f.write('P: '+P+'\n')

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
