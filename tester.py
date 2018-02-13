from lib.feature import *
from lib.crf import CRF
from lib.metric import ErrorAnalyze, ErrorCompare
from sklearn_crfsuite import metrics
from pprint import pprint

train_file = './data/data4.txt'
test_file = './data/test.txt'

train_context = Context(train_file)
test_context = Context(test_file)

train_tdiff = Tdiff(train_file)
test_tdiff = Tdiff(test_file)

train_data = train_context + train_tdiff
test_data = test_context + test_tdiff
crf = CRF()
crf.fit(train_data.X, train_data.Y)
pred_tdiff = crf.predict(test_data.X)
ErrorAnalyze(file_name='pred/tdiff.txt', pred=pred_tdiff, feature_data=test_tdiff)
print(metrics.flat_classification_report(
    test_data.Y, pred_tdiff, labels=('I', 'E'), digits=4
))

train_data = train_context
test_data = test_context
crf = CRF()
crf.fit(train_data.X, train_data.Y)
pred_context = crf.predict(test_data.X)
ErrorAnalyze(file_name='pred/context.txt', pred=pred_context, feature_data=test_context)
print(metrics.flat_classification_report(
    test_data.Y, pred_context, labels=('I', 'E'), digits=4
))

ErrorCompare(file_name='./pred/compare.txt', pred_list=[pred_context, pred_tdiff], feature_data=test_tdiff)
