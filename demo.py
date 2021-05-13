#!/usr/bin/env python
'''
This is a demo. Make sure to run experiment 2 first, so the model will be created, trained and saved.
'''





import tensorflow as tf
import numpy as np
from PIL import Image
from dataset_2 import dataset





input_size = 30
model_name = 'varieties_model'





def main():
    _,_,images,y_test = dataset(input_size)
    x_test = images/255
    model = tf.keras.models.load_model(model_name)
    y_pred = np.argmax(model.predict(x_test),axis=1)
    print('Now the model will classify images. Ctrl-C to exit.')
    for pred,label,image in zip(y_pred,y_test,images):
        print(f'predicted:{pred}, ground truth:{label}')
        Image.fromarray(image).resize((100,100)).show()
        input()






if __name__ == '__main__':
    main()
