#  import pdb
class CompareFile:
    def __init__(self, file_name='pred.txt', line_length=20, pred=None, ans=None, text=None):
        #  pdb.set_trace()
        f = open(file_name, 'w')
        cutter = line_length
        line_true = ''
        line_pred = ''
        for i in range(len(text)):
            word = text[i]
            if i%cutter == 0:
                f.write(line_true+'\n')
                f.write(line_pred+'\n')
                f.write('\n')
                line_true = ''
                line_pred = ''
            line_pred+=word
            line_true+=word
            pred_w = pred[i]
            real_w = ans[i]
            if pred_w == real_w and real_w == 'E':
                line_pred+='，'
                line_true+='，'
            if pred_w == 'E' and real_w == 'I':
                line_pred+='，'
                line_true+='　'
            if pred_w == 'I' and real_w == 'E':
                line_pred+='　'
                line_true+='，'
        f.write(line_true+'\n')
        f.write(line_pred+'\n')
        f.write('\n')
