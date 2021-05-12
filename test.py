#!/usr/bin/env python
'''
This script tests a basic model.
'''

import tensorflow as tf
from dataset import dataset,labels

size = 30

print('### loading dataset')
x_train,y_train,x_test,y_test = dataset(size)

print('### creating model')
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(size,size)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(len(labels))
])

print('### compiling model')
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print('### training model')
model.fit(x_train,y_train,epochs=10)

print('### evaluating model')
print(model.evaluate(x_test,y_test,verbose=2))
