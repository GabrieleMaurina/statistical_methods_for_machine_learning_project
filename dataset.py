#!/usr/bin/env python
'''
This script analyzes the entire fruits folder and saves numpy arrays ready to be
used by a machine learning model.
'''





from os.path import join,basename,isfile
from os import walk
from PIL import Image
import numpy as np





#Dataset folder: change this if you saved the dataset in a different folder
fruits_folder = 'fruits-360'
#Training folder
training_folder = 'Training'
#Testing folder
testing_folder = 'Test'
#Fruits considered
fruits = ['apple','banana','cherry','grape','peach','pear','pepper','plum','potato','tomato']





def find_label(folder):
    '''Find label given folder name.'''
    folder = folder.lower().split(' ')[0]
    for i in range(len(fruits)):
        if folder==fruits[i]:
            return i
    return -1





def get_data(path,size):
    '''Get image data and corresponding labels from folder.'''
    resize_tuple = (size,size)
    for root,dirs,files in walk(path):
        label = find_label(basename(root))
        if label>=0:
            print(root)
            for file in files:
                #load, resize, make grayscale, scale to [0,1]
                image = np.mean(np.asarray(Image.open(join(root,file)).resize(resize_tuple)),axis=2)/255
                yield image,label




def create_dataset_from_folder(path,size):
    '''Create dataset from folder.'''
    data = tuple(zip(*get_data(path)))
    #32 bits is enough since original images are 24 bits
    x = np.array(data[0],dtype=np.float32)
    #8 bits are enough since there are less than 256 label
    y = np.array(data[1],dtype=np.uint8)
    return x,y





def create_dataset(size):
    '''Create dataset.'''
    x_train,y_train = create_dataset_from_folder(join(fruits_folder,training_folder))
    x_test,y_test = create_dataset_from_folder(join(fruits_folder,testing_folder))
    return x_train,y_train,x_test,y_test





def load_dataset(size):
    '''Load dataset from binary files.'''
    x_train = np.load(f'x_train_{size}.npy')
    y_train = np.load(f'y_train_{size}.npy')
    x_test = np.load(f'x_test_{size}.npy')
    y_test = np.load(f'y_test_{size}.npy')
    return x_train,y_train,x_test,y_test





def save_dataset(x_train,y_train,x_test,y_test,size):
    '''Save dataset to binary files for future use.'''
    np.save(f'x_train_{size}.npy',x_train)
    np.save(f'y_train_{size}.npy',y_train)
    np.save(f'x_test_{size}.npy',x_test)
    np.save(f'y_test_{size}.npy',y_test)





def dataset(size):
    files = (
        f'x_train_{size}.npy',
        f'y_train_{size}.npy',
        f'x_test_{size}.npy',
        f'y_test_{size}.npy')
    for f in files:
        if not isfile(f):
            ds = create_dataset(size)
            save_dataset(*ds,size)
            return ds
    return load_dataset(size)
