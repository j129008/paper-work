from lib.data import Data
from lib.learner import Learner
from lib.feature import *
from lib.ensumble_learner import Bagging, Boosting
from lib.metric import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, TimeDistributed, SimpleRNN, Embedding, RNN, GRU, Bidirectional
import unittest

path = './data/data4.txt'
context = Context(path)

class CodeTest(unittest.TestCase):
    def test_mi(self):
        mi = MutualInfo(path)

    def test_tdiff(self):
        tdiff = Tdiff(path)

    def test_rhyme(self):
        rhyme = Rhyme(path, './data/rhyme.txt', './pickles/rhyme_list.pkl', ['反切', '聲母', '韻目', '調', '等', '呼', '韻母'])

    def test_label(self):
        office_data = Label(path, 'office', './ref/known/office2.txt')

    def test_data_adder(self):
        mi = MutualInfo(path)
        data = mi+context
        learner = Learner(data)
        learner.train()
        print('mi+context')
        learner.report()

    def test_vec(self):
        data = UniVec(path)
        x_train, x_test, y_train, y_test = train_test_split(
            data.X, data.Y, test_size=0.6, random_state=1, shuffle=False
        )
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)

        y_pred = clf.predict( x_test )
        y_pred = data.y2lab(y_pred)
        y_test = data.y2lab(y_test)
        print('random forest:')
        print(metrics.flat_classification_report(
            y_test, y_pred, labels=('I', 'E'), digits=4
        ))

    def test_keras(self):
        voc_size = len(set(open(path, 'r').read()))
        k = 1
        data = VecContext(path, k=k)
        x_train, x_test, y_train, y_test = train_test_split(
            data.X, data.Y, test_size=0.6, random_state=1, shuffle=False
        )
        model = Sequential()
        for _ in range(4):
            model.add(Bidirectional(LSTM(50, return_sequences=True, go_backwards=True), input_shape=(k*2+1, 100)))
        model.add(Bidirectional(LSTM(50)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=100, epochs=1)
        pred = model.predict(x_test)
        y_pred = data.y2lab(pred)
        y_test = data.y2lab(y_test)
        print('keras')
        print(metrics.flat_classification_report(
            y_test, y_pred, labels=('I', 'E'), digits=4
        ))

    def test_boost(self):
        context = Context(path)
        context.segment(length=10)
        learner = Boosting(context)
        learner.train()
        print(learner.alpha_list)
        print('boost')
        learner.report()

    def test_bagging(self):
        learner = Bagging(context)
        learner.train()
        print('bagging')
        learner.report()

    def test_learner(self):
        learner = Learner(context)
        learner.train()
        print('crf')
        learner.report()

    def test_CV(self):
        learner = Learner(context)
        learner.train_CV()
        print('crf-CV-tune')
        learner.report()

    def test_analyzer(self):
        learner = Learner(context)
        learner.train()
        pred = learner.predict(learner.X_private)
        ErrorAnalyze(pred=pred, feature_data=learner)
        ErrorCompare(pred_list=[pred, pred], feature_data=learner)

unittest.main()
