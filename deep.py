from lib.feature import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn_crfsuite import metrics
from keras.optimizers import Adam
from keras.layers import LSTM, TimeDistributed, SimpleRNN, Embedding, RNN, GRU, Bidirectional
from sklearn.model_selection import train_test_split

path = './data/data3.txt'
voc_size = len(set(open(path, 'r').read()))
k = 4
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
print(metrics.flat_classification_report(
    y_test, y_pred, labels=('I', 'E'), digits=4
))
