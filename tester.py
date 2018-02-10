from lib.data import Data
from lib.feature import Feature
from lib.crf import CRF
from lib.metric import CompareFile
from sklearn_crfsuite import metrics

train_data = Data('./data/data2.txt')
train_data.load_feature(funcs=[Feature.context], params=[{'k':4, 'n_gram':2}])

test_data = Data('./test.txt')
test_data.load_feature(funcs=[Feature.context], params=[{'k':4, 'n_gram':2}])

crf = CRF()
crf.fit(train_data.X, train_data.Y)
pred = crf.predict(test_data.X)

print(metrics.flat_classification_report(
    test_data.Y, pred, labels=('I', 'E'), digits=4
))

CompareFile(pred=pred, ans=test_data.Y, text=test_data.text)
