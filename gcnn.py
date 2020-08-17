'''
Trains a GDenseNet-40-12 model on the CIFAR-10 Dataset.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras_gcnn.applications.densenetnew import GDenseNet
from dataset import load_dataset
from sklearn.metrics import f1_score


def f1(y_true, y_pred):
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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


batch_size = 64

# nb_classes = 10
##########################################################
nb_classes = 4
##########################################################

epochs = 100

# img_rows, img_cols = 32, 32
# img_channels = 3

##########################################################
img_rows, img_cols = 40, 40
img_channels = 4
##########################################################

# # Parameters for the DenseNet model builder
img_dim = (img_channels, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (
    img_rows, img_cols, img_channels)

depth = 40
nb_dense_block = 3
growth_rate = 3  # number of z2 maps equals growth_rate * group_size, so keep this small.

# nb_filter = 16

##########################################################
nb_filter = 16
##########################################################

##########################################################
# metrics, monitor = ['acc'], 'val_acc'
# metrics, monitor = [f1], 'val_f1'
# metrics, monitor = [f1, 'acc'], 'val_f1'
metrics, monitor = [f1, 'acc'], 'val_acc'
##########################################################

dropout_rate = 0.0  # 0.0 for data augmentation
conv_group = 'D4'  # C4 includes 90 degree rotations, D4 additionally includes reflections in x and y axis.
use_gcnn = True

# # Create the model (without loading weights)
model = GDenseNet(mc_dropout=False, padding='same', nb_dense_block=nb_dense_block, growth_rate=growth_rate,
                  nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None, input_shape=img_dim, depth=depth,
                  use_gcnn=use_gcnn, conv_group=conv_group, classes=nb_classes)
print('Model created')

# model.summary()

optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
print('Finished compiling')

equal_observations = True
##########################################################
(trainX, trainY), (valX, valY), (testX, testY) = load_dataset(img_cols, img_rows, img_channels, equal_observations=equal_observations)
##########################################################

trainX = trainX.astype('float32')
valX = valX.astype('float32')
testX = testX.astype('float32')

# trainX /= 255.
# testX /= 255.

Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_val = np_utils.to_categorical(valY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)


# Test equivariance by comparing outputs for rotated versions of same datapoint:
res = model.predict(np.stack([trainX[123], np.rot90(trainX[123])]))
is_equivariant = np.allclose(res[0], res[1])
print('Equivariance check:', is_equivariant)
assert is_equivariant


generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5. / img_rows,
                               height_shift_range=5. / img_cols)

generator.fit(trainX, seed=0)

##########################################################
weights_file = 'model_weights.h5'
##########################################################

factor = 0.1
min_delta = 0
patience = nb_classes * 4


# lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
#                                cooldown=0, patience=nb_classes, min_lr=0.5e-6)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=factor,
                               cooldown=0, patience=nb_classes, min_lr=0.5e-6)
# early_stopper = EarlyStopping(monitor=f'{monitor}', min_delta=1e-4, patience=nb_classes*2)
early_stopper = EarlyStopping(monitor=f'{monitor}', min_delta=min_delta, patience=patience)
model_checkpoint = ModelCheckpoint(weights_file, monitor=f'{monitor}', save_best_only=True,
                                   save_weights_only=True, mode='auto')

callbacks = [lr_reducer, early_stopper, model_checkpoint]


# print(trainX.shape)
# print(trainY.shape)

# print(valX.shape)
# print(valY.shape)

# print(testX.shape)
# print(testY.shape)

model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=(valX, Y_val),
                    verbose=1)

model.save('full_model.h5')
print('Saved model to disk!')

scores = model.evaluate(testX, Y_test, batch_size=batch_size)

y_pred = model.predict(testX)
y_classes = np.argmax(y_pred, axis=-1)
F1_Scores = f1_score(testY, y_classes, average=None)

print('Test loss : ', scores[0])
print('Test accuracy : ', scores[1])
print('F1 scores : ', F1_Scores)

params = {'batch_size': batch_size, 'nb_classes': nb_classes, 'depth': depth, 
'nb_dense_block': nb_dense_block, 'growth_rate': growth_rate, 'nb_filter': nb_filter,
'dropout_rate': dropout_rate, 'conv_group': conv_group, 'use_gcnn': use_gcnn, 
'factor': factor, 'min_delta': min_delta, 'patience': patience}

text_new = open("model_scores.txt", "a") 
n = text_new.write(f"""
Datashape (x_train, y_train), (x_val, y_val) (x_test, y_test) = {(trainX.shape, trainY.shape), (valX.shape, valY.shape), (testX.shape, testY.shape)}')
Params : {params}
Monitored : {monitor}
Test loss : {scores[0]}
Test accuracy : {scores[1]}
F1 scores : {F1_Scores}
""")
text_new.close()