#!/usr/bin/env python
'''
This is the main experiment.
'''





import tensorflow as tf
import numpy as np
from dataset import dataset,labels
from models import models




output_file = 'results.csv'
input_sizes = [i for i in range(10,51,10)]




def main():
    '''Main experiment.'''

    #write header of csv file
    with open(output_file,'w') as csv:
        csv.write('image_size,model_type,model_depth,model_size,zero_one_loss\n')

    #for 5 input sizes
    for input_size in input_sizes:
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

        #for 30 models
        for model,depth,size,type in models(input_size,len(labels)):
            print(f'### evaluating model {input_size},{type},{depth},{size}')

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

            #fitting a total of 150 models
            #Maximum 50 epochs, but it can stop early based on val_loss
            model.fit(
                x_train,
                y_train,
                epochs=50,
                validation_data=(x_val,y_val),
                callbacks=[early_stopping])

            #compute zero-one loss by counting how many predictions
            #do not match and dividing by number of images
            y_pred = np.argmax(model.predict(x_test),axis=1)
            zero_one_loss =np.sum(y_pred!=y_test)/y_test.shape[0]

            #append data to csv file
            with open(output_file,'a') as csv:
                csv.write(f'{input_size},{type},{depth},{size},{zero_one_loss}\n')





if __name__ == '__main__':
    main()
