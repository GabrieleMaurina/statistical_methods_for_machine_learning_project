#!/usr/bin/env python
'''
This is the main experiment.
'''





import tensorflow as tf
from dataset import dataset,labels
from models import models




output_file = 'results.csv'
input_sizes = [i for i in range(10,51,10)]




def main():
    '''Main experiment.'''

    with open(output_file,'w') as csv:
        #5 input sizes
        for input_size in input_sizes:
            print(f'### loading dataset with size {input_size}')
            x_train,y_train,x_test,y_test = dataset(input_size)
            x_train = x_train/255
            x_test = x_test/255

            #30 models
            for model,depth,size,type in models(input_size,len(labels)):

                #fitting 150 models
                model.fit(x_train,y_train,epochs=5)


                y_pred_binary = tf.round(y_pred)
                temp = tf.cast(tf.equal(y_pred_binary, y_true), tf.float32)
                accuracy = tf.reduce_mean(temp, 1)
                batch_loss = tf.reduce_sum(temp)

                csv.write(f'{input_size},{type},{depth},{size}\n')





if __name__ == '__main__':
    main()
