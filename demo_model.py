#!/usr/bin/env python
'''
This script creates a model for the demo.
'''

import tensorflow as tf
import numpy as np
from os.path import join,isdir,isfile
from os import mkdir
from json import dump
from dataset_2 import dataset
from models import conv




input_size = 30
output_size = 132
depth = 3
size = 40
output_file = 'exp2.csv'
model_name = 'demo_model'




def main():
    print(f'### loading dataset with size {input_size}')
    x_train,y_train,x_test,y_test = dataset(input_size)
    x_train = x_train/255
    x_test = x_test/255

    #10% for validation
    val_size = int(round(y_test.shape[0]*0.1))
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]
    y_train = y_train[val_size:]

    print('### evaluating model')

    model = conv(input_size,output_size,depth,size)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(
        x_train,
        y_train,
        epochs=50,
        validation_data=(x_val,y_val),
        callbacks=[early_stopping])

    n_epochs = len(history.history['loss'])

    #compute zero-one loss by counting how many predictions
    #do not match and dividing by number of images
    y_pred = np.argmax(model.predict(x_test),axis=1)
    zero_one_loss =np.sum(y_pred!=y_test)/y_test.shape[0]

    #create csv file if it does not exist
    if not isfile(output_file):
        with open(output_file,'w') as csv:
            csv.write('image_size,model_type,model_depth,model_size,zero_one_loss,n_epochs\n')

    #append data to csv file
    with open(output_file,'a') as csv:
        csv.write(f'{input_size},conv,{depth},{size},{zero_one_loss},{n_epochs}\n')

    model.save(model_name)





if __name__ == '__main__':
    main()
