from keras.models import load_model
import numpy as np
import os
from keras_gcnn.applications.densenetnew import GDenseNet
from sklearn.metrics import f1_score, precision_recall_fscore_support
from keras.utils import np_utils
from keras import backend as K



def prec_rec_f1(y_true, y_pred, return_all=False):

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    F1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
    if return_all:
        return precision, recall, F1
    return F1

def load_test_dataset(y_dim, x_dim, ch, equal_observations=False, cut_offs=None, seed=None):

    print('Loading dataset...')

    ##########################################################
    test_path = 'CNN_images/test'

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
        num_test_samples = smallest_test*4

    # if we want to slice our data till some cut off index
    if cut_offs:
        smallest_test = int(cut_offs / 4)
        num_test_samples = cut_offs

    # reserve space for all traning, validation and test samples in a numpy array
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
                file_count += smallest_test
            else:
                file_count += len([f for f in os.listdir(test_path + '/' + dirs[dir_count])])

        # list of filenames in directory
        test_filenames = sorted(os.listdir(test_path + '/' + class_dir))

        # for repeatability, use a seed and then shuffle the list of filenames
        np.random.seed(seed)
        np.random.shuffle(test_filenames)

        for j, file_name in enumerate(test_filenames):

            # break if desired number of samples is achieved
            if equal_observations and j == num_test_samples/4:
                break

            file_path = f'{test_path}/{class_dir}/{file_name}'
            # open file 
            with open(file_path, "rb") as file:
                image_array = np.load(file)

                # if we only want rgb images
                if ch == 3:
                    image_array = image_array[:,:,:3]
                image_array = np.moveaxis(image_array, -1, 0)

                # write file to numpy arrays (x_train and y_train)
                (x_test[file_count + j, :, :, :],
                y_test[file_count + j]) = image_array, i

    # convert data to right format
    y_test = np.reshape(y_test, (len(y_test), 1))   

    if K.image_data_format() == 'channels_last':
        x_test = x_test.transpose(0, 2, 3, 1)

    print('loading dataset complete!')
    return (x_test, y_test)


equal_observations = True
seed = 8730
batch_size = 64
nb_classes = 4
Y, X = 40,40

# directory with trained models
model_dir = 'comparisons/models'

# evaluate all configurations of the model
for use_ill_invar in ['ill_invar', 'rgb']:

    for model_used in ['gcnn', 'cnn']:

        # model.summary()

        if use_ill_invar == 'ill_invar':
            img_channels = 4
        elif use_ill_invar == 'rgb':
            img_channels = 3

        # evaluate the model for different (limited) sizes of test data
        for i in range(5):
            if i == 0:
                test_cut = 3750 * 4

            # 75% of 12000
            elif i == 1:
                test_cut = 2812 * 4

            # 50% of 12000
            elif i == 2:
                test_cut = 1875 * 4

            # 25% of 12000
            elif i == 3:
                test_cut = 938 * 4

            # 10% of 12000
            elif i == 4:
                test_cut = 375 * 4

            model = load_model(f'{model_dir}/{model_used}_{use_ill_invar}_{i}_model.h5', custom_objects={'f1':prec_rec_f1})

            # load test images and their corresponding labels.
            x_test, y_test = load_test_dataset(Y, X, img_channels, equal_observations=equal_observations, cut_offs=test_cut, seed=seed)

            # make predictions for each test image
            y_pred = model.predict(x_test)
            # prediction is the class/index with the highest probability
            y_classes = np.argmax(y_pred, axis=-1)


            # precisions = precision(y_test, y_classes)
            # recalls = recall(y_test, y_classes)
            # F1s = f1_score(y_test, y_classes, average=None)

            # calculate precision, recall and F1 score
            precisions, recalls, F1s, _ = precision_recall_fscore_support(y_test, y_classes, average=None)
            
            Y_test = np_utils.to_categorical(y_test, nb_classes)

            scores = model.evaluate(x_test, Y_test, batch_size=batch_size)
            # print('Test loss : ', scores[0])
            # print('Test accuracy : ', scores[1])

            # write results to an image file
            text_new = open(f"comparisons/prec_rec_f1_acc/{model_used}_{use_ill_invar}_{i}_scores.txt", "a") 
            n = text_new.write(f"""
            Datashape (x_test, y_test) = {(x_test.shape, y_test.shape)}')
            img_dimension = {Y, X, img_channels}
            Seed : {seed}

            Test loss : {scores[0]}
            Test accuracy : {scores[1]}

            Recalls : {[r for r in recalls]}
            Precisions :  {[p for p in precisions]}
            F1 scores : {[f for f in F1s]}
            """)
            text_new.close()

# save class predictions
output_path = 'predictions.npy'
with open(output_path, "wb") as f_out:
    np.save(f_out, y_classes)

# save correct class for each image
yTrue_path = 'ytrue.npy' 
with open(yTrue_path, "wb") as f_out:
    np.save(f_out, y_test)

