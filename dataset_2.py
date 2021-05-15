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
dataset_folder = 'dataset_2'





def find_label(folder,labels):
    '''Find label given folder name.'''
    if folder in labels: return labels[folder]
    else:
        n = len(labels)
        print(n)
        labels[folder] = n
        return n





def get_data(path,size,labels):
    '''Get image data and corresponding labels from folder.'''
    resize_tuple = (size,size)
    for root,dirs,files in walk(path):
        if len(files):
            label = find_label(basename(root),labels)
            if label>=0:
                print(root)
                for file in files:
                    #load and resize image
                    image = np.asarray(Image.open(join(root,file)).resize(resize_tuple))
                    yield image,label





def create_dataset_from_folder(path,size,labels):
    '''Create dataset from folder.'''
    data = list(get_data(path,size,labels))
    shuffle(data)
    data = tuple(zip(*data))
    x = np.array(data[0],dtype=np.uint8)
    y = np.array(data[1],dtype=np.uint8)
    return x,y





def create_dataset(size):
    '''Create dataset.'''
    labels = {}
    x_train,y_train = create_dataset_from_folder(join(fruits_folder,training_folder),size,labels)
    x_test,y_test = create_dataset_from_folder(join(fruits_folder,testing_folder),size,labels)
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
    '''Return dataset with images of size.
    If dataset already exists, it is loaded from disk, otherwise it is created
    from the image folder.'''
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
