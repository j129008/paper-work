import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn_crfsuite import metrics
import pickle

# Generate dummy data
x_train = pickle.load(open('./pickles/vec_x_train.pkl','rb'))
#  dim = x_train[0].shape[1]
dim = len(x_train[0])
print(dim)
x_train = np.array(x_train)
x_test  = np.array(pickle.load(open('./pickles/vec_x_test.pkl', 'rb')))
y_test  = np.array(pickle.load(open('./pickles/vec_y_test.pkl','rb')))
y_train = np.array(pickle.load(open('./pickles/vec_y_train.pkl','rb')))

model = Sequential()
model.add(Dense(64, input_dim=dim, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['acc'])
model.fit(x_train, y_train,
                   epochs=2000,
                   batch_size=128)
pred = model.predict(x_test, batch_size=128)

def to_label(y):
    labs = []
    for ele in y:
        if ele > 0.5:
            labs.append('E')
        else:
            labs.append('I')
    return labs

Y_private = y_test
Y_pred = to_label(pred)

print(metrics.flat_classification_report(
    Y_private, Y_pred, labels=('I', 'E'), digits=4
))
