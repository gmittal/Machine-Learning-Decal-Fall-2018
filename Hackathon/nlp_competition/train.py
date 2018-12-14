import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from util import *

train_data = pd.read_csv('train.csv')
x = embed_many(list(train_data['text']))
y = train_data[['start', 'retrieve', 'delete', 'total']].values

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=512))
model.add(Dense(4, activation=tf.nn.softmax))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x, y, shuffle=True, epochs=20)
model.save('save/model.h5')
