class ErrorAnalyze:
    def __init__(self, file_name='pred.txt', pred=None, feature_data=None):
        f = open(file_name, 'w')
        truth = feature_data.Y
        text = feature_data.text
        for i in range(len(text)):
            if pred[i] != truth[i]:
                f.write('{}, T:{}, P:{}, {}\n'.format(text[i], truth[i], pred[i], feature_data.X[i]))
            else:
                if truth[i] == 'E':
                    f.write('{}, {}\n\n'.format(text[i], feature_data.X[i]))
                else:
                    f.write('{}, {}\n'.format(text[i], feature_data.X[i]))
