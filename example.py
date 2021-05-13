#!/usr/bin/env python
'''
This script tests a basic model.
'''

import tensorflow as tf
import tensorflow_docs as tfdocs
import numpy as np
from dataset import dataset,labels

size = 30

print('### loading dataset')
x_train,y_train,x_test,y_test = dataset(size)
x_train = x_train/255
x_test = x_test/255

print('### creating model')
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(size,size,3)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(len(labels))
])

print('### compiling model')
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print('### training model')
history = model.fit(x_train,y_train,epochs=4)

plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy',smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])

exit(0)

print('### evaluating model')
print(model.evaluate(x_test,y_test,verbose=2))

print('### zero one loss')
y_pred = np.argmax(model.predict(x_test),axis=1)
zero_one_loss =np.sum(y_pred!=y_test)/y_test.shape[0]
print(zero_one_loss)
print(y_pred)
print(y_test)
