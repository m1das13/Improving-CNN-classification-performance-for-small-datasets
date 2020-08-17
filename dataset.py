x`"""CIFAR10 small images classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
import numpy as np
import os



def load_dataset(y_dim, x_dim, ch, equal_observations=False, cut_offs=None, seed=None):
    # Usage:
    # - Input:
    # y_dim: vertical dimension of input and output image. Must be of type int.
    # x_dim: horizontal dimension of input and output image. Must be of type int.
    # ch: number of desired output channels (3 = rgb, 4 = rgb + illumination invariance). Must be either 3 or 4 and of type int. 
    # equal_observations: if we want our train, validation and test size to be of an equal amount of observations. Must be of type bool.
    # cut_offs: if we want to slice our training, validation of test data till some 'cut off' index. Must be of type int.
    # seed: specify seed for repeatable randomization of directories. Must be of size int.
    # - Output:
    # Three tuples containing the numpy arrays for the training, validation and test images


    print('Loading dataset...')

    ##########################################################
    train_path = 'CNN_images/train'
    val_path = 'CNN_images/val'
    test_path = 'CNN_images/test'

    ##########################################################

    smallest_train = 99999999

    # count number of train samples
    num_train_samples = 0
    for class_dir in os.listdir(train_path):
        dir_size = len([f for f in os.listdir(train_path + '/' + class_dir)])
        num_train_samples += dir_size
        # print(dir_size)

        if equal_observations and dir_size < smallest_train:
            smallest_train = dir_size

    smallest_val = 99999999

    # count number of validation samples
    num_val_samples = 0
    for class_dir in os.listdir(val_path):
        dir_size = len([f for f in os.listdir(val_path + '/' + class_dir)])
        num_val_samples += dir_size
        # print(dir_size)

        if equal_observations and dir_size < smallest_val:
            smallest_val = dir_size

    smallest_test = 99999999

    # count number of test samples
    num_test_samples = 0
    for class_dir in os.listdir(test_path):
        dir_size = len([f for f in os.listdir(test_path + '/' + class_dir)])
        num_test_samples += dir_size
        # print(dir_size)

        if equal_observations and dir_size < smallest_test:
            smallest_test = dir_size

    ##########################################################
    # if we want our train, validation and test size to be of an equal amount of observations
    if equal_observations:
        num_train_samples = smallest_train*4
        num_val_samples = smallest_val*4
        num_test_samples = smallest_test*4

    # if we want to slice our data till some cut off index
    if cut_offs:
        smallest_train = int(cut_offs[0] / 4)
        smallest_val = int(cut_offs[1] / 4)
        smallest_test = int(cut_offs[2] / 4)

        num_train_samples = cut_offs[0]
        num_val_samples = cut_offs[1]
        num_test_samples = cut_offs[2]

    ##########################################################

    # reserve space for all traning, validation and test samples in a numpy array
    x_train = np.empty((num_train_samples, ch, y_dim, x_dim), dtype='float32')
    y_train = np.empty((num_train_samples,), dtype='float32')

    x_val = np.empty((num_val_samples, ch, y_dim, x_dim), dtype='float32')
    y_val = np.empty((num_val_samples,), dtype='float32')

    x_test = np.empty((num_test_samples, ch, y_dim, x_dim), dtype='float32')
    y_test = np.empty((num_test_samples,), dtype='float32')

    # print(f'Datashape (x_train, y_train), (x_val, y_val) (x_test, y_test) = {(x_train.shape, y_train.shape), (x_val.shape, y_val.shape), (x_test.shape, y_test.shape)}')

    ##########################################################

    dirs = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']


    for i, class_dir in enumerate(dirs):
        file_count = 0
        # iterate over the four directories and determine the amount of files in it
        for dir_count in range(0, i):
            # if we want equal observations for each class we limit the number of files to the class with the smallest amount of data (i.e. smallest training data)
            if equal_observations:
                file_count += smallest_train
            else:
                file_count += len([f for f in os.listdir(train_path + '/' + dirs[dir_count])])

        # list of filenames in directory
        train_filenames = sorted(os.listdir(train_path + '/' + class_dir))
        
        # for repeatability, use a seed and then shuffle the list of filenames
        np.random.seed(seed)
        np.random.shuffle(train_filenames)
        
        for j, file_name in enumerate(train_filenames):

            # break if desired number of samples is achieved
            if equal_observations and j == num_train_samples/4:
                break

            file_path = f'{train_path}/{class_dir}/{file_name}'
            # open file 
            with open(file_path, "rb") as file:
                image_array = np.load(file)

                # if we only want rgb images
                if ch == 3:
                    image_array = image_array[:,:,:3]

                image_array = np.moveaxis(image_array, -1, 0)

                # write file to numpy arrays (x_train and y_train)
                (x_train[file_count + j, :, :, :],
                y_train[file_count + j]) = image_array, i


    # repeat process for validation data
    for i, class_dir in enumerate(dirs):
        file_count = 0
        for dir_count in range(0, i):
            if equal_observations:
                file_count += smallest_val
            else:
                file_count += len([f for f in os.listdir(val_path + '/' + dirs[dir_count])])

        val_filenames = sorted(os.listdir(val_path + '/' + class_dir))

        np.random.seed(seed)
        np.random.shuffle(val_filenames)

        for j, file_name in enumerate(val_filenames):
            if equal_observations and j == num_val_samples/4:

                break
            file_path = f'{val_path}/{class_dir}/{file_name}'

            with open(file_path, "rb") as file:
                image_array = np.load(file)
                if ch == 3:
                    image_array = image_array[:,:,:3]
                image_array = np.moveaxis(image_array, -1, 0)
                (x_val[file_count + j, :, :, :],
                y_val[file_count + j]) = image_array, i


    # repeat process for validation data
    for i, class_dir in enumerate(dirs):
        file_count = 0
        for dir_count in range(0, i):
            if equal_observations:
                file_count += smallest_test
            else:
                file_count += len([f for f in os.listdir(test_path + '/' + dirs[dir_count])])

        test_filenames = sorted(os.listdir(test_path + '/' + class_dir))

        np.random.seed(seed)
        np.random.shuffle(test_filenames)

        for j, file_name in enumerate(test_filenames):
            if equal_observations and j == num_test_samples/4:
                break
            file_path = f'{test_path}/{class_dir}/{file_name}'

            with open(file_path, "rb") as file:
                image_array = np.load(file)
                if ch == 3:
                    image_array = image_array[:,:,:3]
                image_array = np.moveaxis(image_array, -1, 0)
                (x_test[file_count + j, :, :, :],
                y_test[file_count + j]) = image_array, i


    ##########################################################

    # convert data to right format
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_val = np.reshape(y_val, (len(y_val), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))   
    

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_val = x_val.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    print('loading dataset complete!')

    # print(f'Datashape (x_train, y_train), (x_val, y_val) (x_test, y_test) = {(x_train.shape, y_train.shape), (x_val.shape, y_val.shape), (x_test.shape, y_test.shape)}')
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# x_dim = 50
# y_dim = 50
# channels = 4

# load_dataset(y_dim, x_dim, channels)