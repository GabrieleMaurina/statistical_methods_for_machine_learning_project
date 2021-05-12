#!/usr/bin/env python
'''
Main script.
'''

import tensorflow as tf
from dataset import dataset

print('### loading dataset')

x_train,y_train,x_test,y_test = dataset(30)


'''print('### creating model')
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print(f'### predictions = {predictions}')

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print(loss_fn(y_train[:1], predictions).numpy())

print('### compiling model')
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

print('### fitting model')
model.fit(x_train, y_train, epochs=5)

print('### evaluating model')
print(model.evaluate(x_test,  y_test, verbose=2))

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print(probability_model(x_test[:5]))'''
