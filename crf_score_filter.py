import pickle
import re
from collections import Counter
from sklearn_crfsuite import metrics

crf  = pickle.load(open('./pickles/crf_2.pkl', 'rb'))
X    = pickle.load(open('./pickles/X_private.pkl', 'rb'))
Y    = pickle.load(open('./pickles/Y_private.pkl', 'rb'))
prob_x = crf.predict_marginals([ X ])[0]


gap_list = [-1.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for gap in gap_list:
    print("gap:", ( 1.0 + gap )/2)
    high_score_y = []
    high_score_pred_y = []
    for i in range(len( prob_x )):
        score = prob_x[i]['E'] - prob_x[i]['I']
        if abs( score ) > gap:
            high_score_y.append(Y[i])
            if score > 0 :
                high_score_pred_y.append('E')
            else:
                high_score_pred_y.append('I')

    print( "high score rate(all): " , len(high_score_y)/len(Y) )
    print( "high score rate(E): " , Counter(high_score_y)['E']/Counter(Y)['E'])
    print( "high score rate(I): " , Counter(high_score_y)['I']/Counter(Y)['I'])

    print(metrics.flat_classification_report(
        high_score_y, high_score_pred_y, labels=['I','E'], digits=3
    ))
