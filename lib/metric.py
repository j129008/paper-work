class ErrorAnalyze:
    def __init__(self, file_name='pred.txt', pred=None, feature_data=None):
        f = open(file_name, 'w')
        truth = feature_data.Y
        text = feature_data.text
        for i in range(len(text)):
            if pred[i] != truth[i]:
                f.write('{}, T:{}, P:{}, {}\n'.format(text[i], truth[i], pred[i], feature_data.X[i]))
            else:
                f.write('{}, {}\n'.format(text[i], feature_data.X[i]))
            if truth[i] == 'E':
                f.write('\n')

class ErrorCompare:
    def __init__(self, file_name='compare.txt', pred_list=None, feature_data=None):
        f = open(file_name, 'w')
        text = feature_data.text
        truth = feature_data.Y
        for i in range(len(text)):
            pred_i = [truth[i]] + [ pred[i] for pred in pred_list]
            if len(set(pred_i)) != 1:
                f.write('{}, {}, {}\n'.format(text[i], pred_i[1:], feature_data.X[i]))
            else:
                f.write('{}\n'.format(text[i]) )
            if truth[i] == 'E':
                f.write('\n')
