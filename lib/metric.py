import pdb
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
