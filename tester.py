from lib.data import Data
from lib.feature import *
from lib.crf import CRF
from lib.metric import CompareFile
from sklearn_crfsuite import metrics

path = './data/data4.txt'
train_data = Context(path) + MutualInfo(path)
test_data = Context('./test.txt')

crf = CRF()
crf.fit(train_data.X, train_data.Y)
pred = crf.predict(test_data.X)

print(metrics.flat_classification_report(
    test_data.Y, pred, labels=('I', 'E'), digits=4
))

CompareFile(pred=pred, ans=test_data.Y, text=test_data.text)
