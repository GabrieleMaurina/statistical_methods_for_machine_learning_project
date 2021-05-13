'''
Dataset utility module.
Make sure to download the fruits-360 dataset first. Then copy it inside this directory.
'''





from os.path import join,basename,isfile,isdir
from os import walk,mkdir
from PIL import Image
import numpy as np
from random import shuffle





#Images folder: change this if you saved the dataset in a different folder
fruits_folder = 'fruits-360'
#Training folder
training_folder = 'Training'
#Testing folder
testing_folder = 'Test'
#Save folder
dataset_folder = 'dataset_folder'
#Fruits considered
labels = ['apple','banana','cherry','grape','peach','pear','pepper','plum','potato','tomato']





def find_label(folder):
    '''Find label given folder name.'''
    folder = folder.lower().split(' ')[0]
    for i in range(len(labels)):
        if folder==labels[i]:
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
                #load and resize image
                image = np.asarray(Image.open(join(root,file)).resize(resize_tuple))
                yield image,label




def create_dataset_from_folder(path,size):
    '''Create dataset from folder.'''
    data = list(get_data(path,size))
    shuffle(data)
    data = tuple(zip(*data))
    x = np.array(data[0],dtype=np.uint8)
    y = np.array(data[1],dtype=np.uint8)
    return x,y





def create_dataset(size):
    '''Create dataset.'''
    x_train,y_train = create_dataset_from_folder(join(fruits_folder,training_folder),size)
    x_test,y_test = create_dataset_from_folder(join(fruits_folder,testing_folder),size)
    return x_train,y_train,x_test,y_test





def load_dataset(size):
    '''Load dataset from binary files.'''
    x_train = np.load(join(dataset_folder,f'x_train_{size}.npy'))
    y_train = np.load(join(dataset_folder,f'y_train_{size}.npy'))
    x_test = np.load(join(dataset_folder,f'x_test_{size}.npy'))
    y_test = np.load(join(dataset_folder,f'y_test_{size}.npy'))
    return x_train,y_train,x_test,y_test





def save_dataset(x_train,y_train,x_test,y_test,size):
    '''Save dataset to binary files for future use.'''
    if not isdir(dataset_folder):
        mkdir(dataset_folder)
    np.save(join(dataset_folder,f'x_train_{size}.npy'),x_train)
    np.save(join(dataset_folder,f'y_train_{size}.npy'),y_train)
    np.save(join(dataset_folder,f'x_test_{size}.npy'),x_test)
    np.save(join(dataset_folder,f'y_test_{size}.npy'),y_test)





def dataset(size):
    files = (
        join(dataset_folder,f'x_train_{size}.npy'),
        join(dataset_folder,f'y_train_{size}.npy'),
        join(dataset_folder,f'x_test_{size}.npy'),
        join(dataset_folder,f'y_test_{size}.npy'))
    for f in files:
        if not isfile(f):
            ds = create_dataset(size)
            save_dataset(*ds,size)
            return ds
    return load_dataset(size)
