'''
Models utility module.
'''





import tensorflow as tf





def dense(input_size,output_size,depth,size):
    '''Create a dense model with specific input_size,output_size,depth and number of neuros.'''
    layers = [tf.keras.layers.Flatten(input_shape=(input_size,input_size,3))]
    for i in range(depth):
        layers.append(tf.keras.layers.Dense(size,activation='relu'))
    layers.append(tf.keras.layers.Dense(output_size))
    return tf.keras.Sequential(layers)





def conv(input_size,output_size,depth,size):
    '''Create a conv model with specific input_size,output_size,depth and number of neuros.'''
    layers = [tf.keras.layers.Conv2D(size,(3, 3),activation='relu',input_shape=(input_size,input_size,3))]
    for i in range(depth-1):
        layers += [
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(size,(3, 3),activation='relu',padding='same')]
    layers += [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(size,activation='relu'),
        tf.keras.layers.Dense(output_size)]
    return tf.keras.Sequential(layers)





def models(input_size,output_size):
    '''This generator returns models to test in the experiment.'''


    #dense layers, different sizes
    for i in range(1,4):
        for j in range(1,6):
            yield dense(input_size,output_size,i,j*32),i,j*32,'dense'


    #conv model, different sizes
    for i in range(1,4):
        for j in range(1,6):
            yield conv(input_size,output_size,i,j*8),i,j*8,'conv'
