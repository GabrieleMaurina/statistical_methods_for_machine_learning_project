#!/usr/bin/env python
'''
This script tests a basic convolutional model.
'''

import tensorflow as tf
from dataset import dataset,labels

size = 30

print('### loading dataset')
x_train,y_train,x_test,y_test = dataset(size)
x_train = x_train/255
x_test = x_test/255

print('### creating model')
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3, 3),activation='relu',input_shape=(size,size,3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32,(3, 3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32,(3, 3),activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(len(labels))
])

model.summary()
exit(0)

print('### compiling model')
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print('### training model')
model.fit(x_train,y_train,epochs=5)

print('### evaluating model')
print(model.evaluate(x_test,y_test,verbose=2))
