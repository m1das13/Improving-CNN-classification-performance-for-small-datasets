'''
Trains a GDenseNet-40-12 model on the CIFAR-10 Dataset.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
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


# seed = np.random.randint(0, 10000)
seed = 8730

# compare g-cnn with cnn
for use_ill_invar in ['rgb', 'ill_invar']:
    # print()

    for i in range(1, 5):
        equal_observations = True
        nb_classes = 4
        img_rows, img_cols = 96, 96

        ##########################################################
        if use_ill_invar == 'ill_invar':
            img_channels = 4
        elif use_ill_invar == 'rgb':
            img_channels = 3
        ##########################################################
        ##########################################################
        if i == 0:
            train_cut = 12000 * 4
            val_cut = 3000 * 4
            test_cut = 3750 * 4

        # 75% of 12000
        elif i == 1:
            train_cut = 9000 * 4
            val_cut = 2250 * 4
            test_cut = 2812 * 4

        # 50% of 12000
        elif i == 2:
            train_cut = 6000 * 4
            val_cut = 1500 * 4
            test_cut = 1875 * 4

        # 25% of 12000
        elif i == 3:
            train_cut = 3000 * 4
            val_cut = 750 * 4
            test_cut = 938 * 4

        # 10% of 12000
        elif i == 4:
            train_cut = 1200 * 4
            val_cut = 300 * 4
            test_cut = 375 * 4

        # 5% of 12000
        elif i == 5:
            train_cut = 600 * 4
            val_cut = 150 * 4
            test_cut = 188 * 4

        # 2.5% of 12000s
        elif i == 6:
            train_cut = 300 * 4
            val_cut = 75 * 4
            test_cut = 94 * 4

        # 1% of 12000s
        elif i == 7:
            train_cut = 120 * 4
            val_cut = 30 * 4
            test_cut = 38 * 4
        


        # cut_offs = [train_cut, val_cut, test_cut]
        # (trainX, trainY), (valX, valY), (testX, testY) = load_dataset(img_cols, img_rows, img_channels, equal_observations=equal_observations, cut_offs=cut_offs, seed=seed)

        # trainX = trainX.astype('float32')
        # valX = valX.astype('float32')
        # testX = testX.astype('float32')

        # Y_train = np_utils.to_categorical(trainY, nb_classes)
        # Y_val = np_utils.to_categorical(valY, nb_classes)
        # Y_test = np_utils.to_categorical(testY, nb_classes)

        trainX = np.zeros((400, img_cols,img_rows,3))
        valX = np.zeros((100, img_cols,img_rows,3))
        testX = np.zeros((100, img_cols,img_rows,3))

        trainY = np.zeros((400,))
        valY = np.zeros((100,))
        testY = np.zeros((100,))

        Y_train = np_utils.to_categorical(trainY, nb_classes)
        Y_val = np_utils.to_categorical(valY, nb_classes)
        Y_test = np_utils.to_categorical(testY, nb_classes)
        #########################################################
        
        # print(f'Datashape (x_train, y_train), (x_val, y_val) (x_test, y_test) = {(trainX.shape, trainY.shape), (valX.shape, valY.shape), (testX.shape, testY.shape)}')

        for model_used in ['gcnn', 'cnn']:
            if model_used == 'gcnn':
                use_gcnn = True
            elif model_used == 'cnn':
                use_gcnn = False


            batch_size = 64
            epochs = 50

            # # Parameters for the DenseNet model builder
            img_dim = (img_channels, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (
                img_rows, img_cols, img_channels)

            depth = 40
            nb_dense_block = 3
            growth_rate = 3  # number of z2 maps equals growth_rate * group_size, so keep this small.
            nb_filter = 16

            # metrics, monitor = ['acc'], 'val_acc'
            # metrics, monitor = [f1], 'val_f1'
            # metrics, monitor = [f1, 'acc'], 'val_f1'
            metrics, monitor = [f1, 'acc'], 'val_acc'
            ##########################################################

            dropout_rate = 0.0  # 0.0 for data augmentation
            conv_group = 'C4'  # C4 includes 90 degree rotations, D4 additionally includes reflections in x and y axis.
            

            # Create the model (without loading weights)
            model = GDenseNet(mc_dropout=False, padding='same', nb_dense_block=nb_dense_block, growth_rate=growth_rate,
                            nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None, input_shape=img_dim, depth=depth,
                            use_gcnn=use_gcnn, conv_group=conv_group, classes=nb_classes)
            print('Model created')
            model.summary()
            optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
            print('Finished compiling')

            # Test equivariance by comparing outputs for rotated versions of same datapoint:
            res = model.predict(np.stack([trainX[123], np.rot90(trainX[123])]))
            is_equivariant = np.allclose(res[0], res[1])
            print('Equivariance check:', is_equivariant)
            assert is_equivariant


            generator = ImageDataGenerator(rotation_range=15,
                                        width_shift_range=5. / img_rows,
                                        height_shift_range=5. / img_cols)

            generator.fit(trainX, seed=0)

            # ##########################################################
            weights_file = 'model_weights.h5'
            # ##########################################################

            factor = 0.1
            min_delta = 0
            patience = nb_classes * 4

            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=factor,
                                        cooldown=0, patience=nb_classes, min_lr=0.5e-6)
            # early_stopper = EarlyStopping(monitor=f'{monitor}', min_delta=min_delta, patience=patience)
            model_checkpoint = ModelCheckpoint(weights_file, monitor=f'{monitor}', save_best_only=True,
                                            save_weights_only=True, mode='auto')

            # callbacks = [lr_reducer, early_stopper, model_checkpoint]
            callbacks = [lr_reducer, model_checkpoint]

            model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size,
                                epochs=epochs,
                                callbacks=callbacks,
                                validation_data=(valX, Y_val),
                                verbose=1)
            # print(f'comparisons/models/{model_used}_{use_ill_invar}_{i}_model.h5')
            model.save(f'comparisons/models/{model_used}_{use_ill_invar}_{i}_model.h5')
            print('Saved model to disk!')

            scores = model.evaluate(testX, Y_test, batch_size=batch_size)

            y_pred = model.predict(testX)
            y_classes = np.argmax(y_pred, axis=-1)
            F1_Scores = f1_score(testY, y_classes, average=None)

            print('Test loss : ', scores[0])
            print('Test accuracy : ', scores[1])
            print('F1 scores : ', F1_Scores)

            params = {'use_ill_invar': use_ill_invar, 'use_gcnn': use_gcnn, 'train_size': train_cut, 
            'val_size': val_cut, 'test_size': test_cut, 'conv_group': conv_group}
            # print(f"comparisons/results/{model_used}_{use_ill_invar}_{i}_scores.txt")
            text_new = open(f"comparisons/results/{model_used}_{use_ill_invar}_{i}_scores.txt", "a") 
            n = text_new.write(f"""
            Datashape (x_train, y_train), (x_val, y_val) (x_test, y_test) = {(trainX.shape, trainY.shape), (valX.shape, valY.shape), (testX.shape, testY.shape)}')
            img_dimension = {img_rows, img_cols, img_channels}
            Seed : {seed}
            Params : {params}
            Monitored : {monitor}
            Test loss : {scores[0]}
            Test accuracy : {scores[1]}
            F1 scores : {F1_Scores}
            """)
            text_new.close()





# # compare g-cnn with cnn
# for i in range(1,2):
#     equal_observations = True
#     nb_classes = 4

#     ##########################################################
#     img_rows, img_cols = 40, 40
#     img_channels = 4
#     ##########################################################

#     if i == 1:
#         use_gcnn = True
#         model_used = 'gcnn'
#     elif i == 0:
#         use_gcnn = False
#         model_used = 'cnn'

#     use_ill_invar = 'ill_invar'
#     # print()
#     # print(model_used)
#     ##########################################################

#     for j in range(5):
#         if i == 0: 
#             if j == 4:
#                 break
#             j += 1
            
#         ##########################################################
#         if j == 0:
#             train_cut = 12000 * 4
#             val_cut = 3000 * 4
#             test_cut = 3750 * 4

#         elif j == 1:
#             train_cut = 9000 * 4
#             val_cut = 2250 * 4
#             test_cut = 2812 * 4

#         elif j == 2:
#             train_cut = 6750 * 4
#             val_cut = 1687 * 4
#             test_cut = 2109 * 4

#         elif j == 3:
#             train_cut = 5062 * 4
#             val_cut = 1265 * 4
#             test_cut = 1582 * 4

#         elif j == 4:
#             train_cut = 3796 * 4
#             val_cut = 949 * 4
#             test_cut = 1186 * 4

#         # print(f'{model_used}_{use_ill_invar}_{j}')
#         cut_offs = [train_cut, val_cut, test_cut]
#         (trainX, trainY), (valX, valY), (testX, testY) = load_dataset(img_cols, img_rows, img_channels, equal_observations=equal_observations, cut_offs=cut_offs)

#         trainX = trainX.astype('float32')
#         valX = valX.astype('float32')
#         testX = testX.astype('float32')

#         Y_train = np_utils.to_categorical(trainY, nb_classes)
#         Y_val = np_utils.to_categorical(valY, nb_classes)
#         Y_test = np_utils.to_categorical(testY, nb_classes)
#         #########################################################

#         batch_size = 64
#         epochs = 50

#         # # Parameters for the DenseNet model builder
#         img_dim = (img_channels, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (
#             img_rows, img_cols, img_channels)

#         depth = 40
#         nb_dense_block = 3
#         growth_rate = 3  # number of z2 maps equals growth_rate * group_size, so keep this small.
#         nb_filter = 16

#         # metrics, monitor = ['acc'], 'val_acc'
#         # metrics, monitor = [f1], 'val_f1'
#         # metrics, monitor = [f1, 'acc'], 'val_f1'
#         metrics, monitor = [f1, 'acc'], 'val_acc'
#         ##########################################################

#         dropout_rate = 0.0  # 0.0 for data augmentation
#         conv_group = 'D4'  # C4 includes 90 degree rotations, D4 additionally includes reflections in x and y axis.
        

#         # Create the model (without loading weights)
#         model = GDenseNet(mc_dropout=False, padding='same', nb_dense_block=nb_dense_block, growth_rate=growth_rate,
#                         nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None, input_shape=img_dim, depth=depth,
#                         use_gcnn=use_gcnn, conv_group=conv_group, classes=nb_classes)
#         print('Model created')
#         # model.summary()
#         optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
#         model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
#         print('Finished compiling')

#         # Test equivariance by comparing outputs for rotated versions of same datapoint:
#         # res = model.predict(np.stack([trainX[123], np.rot90(trainX[123])]))
#         # is_equivariant = np.allclose(res[0], res[1])
#         # print('Equivariance check:', is_equivariant)
#         # assert is_equivariant


#         generator = ImageDataGenerator(rotation_range=15,
#                                     width_shift_range=5. / img_rows,
#                                     height_shift_range=5. / img_cols)

#         generator.fit(trainX, seed=0)

#         # ##########################################################
#         weights_file = 'model_weights.h5'
#         # ##########################################################

#         factor = 0.1
#         min_delta = 0
#         patience = nb_classes * 4

#         lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=factor,
#                                     cooldown=0, patience=nb_classes, min_lr=0.5e-6)
#         # early_stopper = EarlyStopping(monitor=f'{monitor}', min_delta=min_delta, patience=patience)
#         model_checkpoint = ModelCheckpoint(weights_file, monitor=f'{monitor}', save_best_only=True,
#                                         save_weights_only=True, mode='auto')

#         # callbacks = [lr_reducer, early_stopper, model_checkpoint]
#         callbacks = [lr_reducer, model_checkpoint]

#         model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size,
#                             epochs=epochs,
#                             callbacks=callbacks,
#                             validation_data=(valX, Y_val),
#                             verbose=1)
#         model.save(f'comparisons/gcnn_vs_cnn/model/{model_used}_{use_ill_invar}_{j}_model.h5')
#         print('Saved model to disk!')

#         scores = model.evaluate(testX, Y_test, batch_size=batch_size)

#         y_pred = model.predict(testX)
#         y_classes = np.argmax(y_pred, axis=-1)
#         F1_Scores = f1_score(testY, y_classes, average=None)

#         print('Test loss : ', scores[0])
#         print('Test accuracy : ', scores[1])
#         print('F1 scores : ', F1_Scores)

#         params = {'use_ill_invar': use_ill_invar, 'use_gcnn': use_gcnn, 'train_size': train_cut, 
#         'val_size': val_cut, 'test_size': test_cut, 'conv_group': conv_group}

#         text_new = open(f"comparisons/gcnn_vs_cnn/{model_used}_{use_ill_invar}_{j}_scores.txt", "a") 
#         n = text_new.write(f"""
#         Datashape (x_train, y_train), (x_val, y_val) (x_test, y_test) = {(trainX.shape, trainY.shape), (valX.shape, valY.shape), (testX.shape, testY.shape)}')
#         img_dimension = {img_rows, img_cols, img_channels}
#         Params : {params}
#         Monitored : {monitor}
#         Test loss : {scores[0]}
#         Test accuracy : {scores[1]}
#         F1 scores : {F1_Scores}
#         """)
#         text_new.close()


# # compare ill_invar with rgb with cnn
# for i in range(1, 2):
#     equal_observations = True
#     nb_classes = 4

#     ##########################################################
#     img_rows, img_cols = 40, 40

#     if i == 0:
#         img_channels = 4
#         use_ill_invar = 'ill_invar'

#     elif i == 1:
#         img_channels = 3
#         use_ill_invar = 'rgb'

#     ##########################################################
    


#     ##########################################################
#     use_gcnn = False
#     model_used = 'cnn'
#     # print()
#     # print(model_used)
#     ##########################################################

#     for j in range(5):
        
#         ##########################################################
#         if j == 0:
#             train_cut = 12000 * 4
#             val_cut = 3000 * 4
#             test_cut = 3750 * 4

#         elif j == 1:
#             train_cut = 9000 * 4
#             val_cut = 2250 * 4
#             test_cut = 2812 * 4

#         elif j == 2:
#             train_cut = 6750 * 4
#             val_cut = 1687 * 4
#             test_cut = 2109 * 4

#         elif j == 3:
#             train_cut = 5062 * 4
#             val_cut = 1265 * 4
#             test_cut = 1582 * 4

#         elif j == 4:
#             train_cut = 3796 * 4
#             val_cut = 949 * 4
#             test_cut = 1186 * 4

#         # print(f'{model_used}_{use_ill_invar}_{j}')
#         cut_offs = [train_cut, val_cut, test_cut]
#         (trainX, trainY), (valX, valY), (testX, testY) = load_dataset(img_cols, img_rows, img_channels, equal_observations=equal_observations, cut_offs=cut_offs)

#         trainX = trainX.astype('float32')
#         valX = valX.astype('float32')
#         testX = testX.astype('float32')

#         Y_train = np_utils.to_categorical(trainY, nb_classes)
#         Y_val = np_utils.to_categorical(valY, nb_classes)
#         Y_test = np_utils.to_categorical(testY, nb_classes)
#         #########################################################

#         batch_size = 64
#         epochs = 50

#         # # Parameters for the DenseNet model builder
#         img_dim = (img_channels, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (
#             img_rows, img_cols, img_channels)

#         depth = 40
#         nb_dense_block = 3
#         growth_rate = 3  # number of z2 maps equals growth_rate * group_size, so keep this small.
#         nb_filter = 16

#         # metrics, monitor = ['acc'], 'val_acc'
#         # metrics, monitor = [f1], 'val_f1'
#         # metrics, monitor = [f1, 'acc'], 'val_f1'
#         metrics, monitor = [f1, 'acc'], 'val_acc'
#         ##########################################################

#         dropout_rate = 0.0  # 0.0 for data augmentation
#         conv_group = 'D4'  # C4 includes 90 degree rotations, D4 additionally includes reflections in x and y axis.
        

#         # Create the model (without loading weights)
#         model = GDenseNet(mc_dropout=False, padding='same', nb_dense_block=nb_dense_block, growth_rate=growth_rate,
#                         nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None, input_shape=img_dim, depth=depth,
#                         use_gcnn=use_gcnn, conv_group=conv_group, classes=nb_classes)
#         print('Model created')
#         # model.summary()
#         optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
#         model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
#         print('Finished compiling')

#         # Test equivariance by comparing outputs for rotated versions of same datapoint:
#         # res = model.predict(np.stack([trainX[123], np.rot90(trainX[123])]))
#         # is_equivariant = np.allclose(res[0], res[1])
#         # print('Equivariance check:', is_equivariant)
#         # assert is_equivariant


#         generator = ImageDataGenerator(rotation_range=15,
#                                     width_shift_range=5. / img_rows,
#                                     height_shift_range=5. / img_cols)

#         generator.fit(trainX, seed=0)

#         # ##########################################################
#         weights_file = 'model_weights.h5'
#         # ##########################################################

#         factor = 0.1
#         min_delta = 0
#         patience = nb_classes * 4

#         lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=factor,
#                                     cooldown=0, patience=nb_classes, min_lr=0.5e-6)
#         # early_stopper = EarlyStopping(monitor=f'{monitor}', min_delta=min_delta, patience=patience)
#         model_checkpoint = ModelCheckpoint(weights_file, monitor=f'{monitor}', save_best_only=True,
#                                         save_weights_only=True, mode='auto')

#         # callbacks = [lr_reducer, early_stopper, model_checkpoint]
#         callbacks = [lr_reducer, model_checkpoint]

#         model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size,
#                             epochs=epochs,
#                             callbacks=callbacks,
#                             validation_data=(valX, Y_val),
#                             verbose=1)

#         model.save(f'comparisons/cnn_ill_invar_vs_rgb/model/{model_used}_{use_ill_invar}_{j}_model.h5')
#         print('Saved model to disk!')

#         scores = model.evaluate(testX, Y_test, batch_size=batch_size)

#         y_pred = model.predict(testX)
#         y_classes = np.argmax(y_pred, axis=-1)
#         F1_Scores = f1_score(testY, y_classes, average=None)

#         print('Test loss : ', scores[0])
#         print('Test accuracy : ', scores[1])
#         print('F1 scores : ', F1_Scores)

#         params = {'use_ill_invar': use_ill_invar, 'use_gcnn': use_gcnn, 'train_size': train_cut, 
#         'val_size': val_cut, 'test_size': test_cut, 'conv_group': conv_group}

#         text_new = open(f"comparisons/cnn_ill_invar_vs_rgb/{model_used}_{use_ill_invar}_{j}_scores.txt", "a") 
#         n = text_new.write(f"""
#         Datashape (x_train, y_train), (x_val, y_val) (x_test, y_test) = {(trainX.shape, trainY.shape), (valX.shape, valY.shape), (testX.shape, testY.shape)}')
#         img_dimension = {img_rows, img_cols, img_channels}
#         Params : {params}
#         Monitored : {monitor}
#         Test loss : {scores[0]}
#         Test accuracy : {scores[1]}
#         F1 scores : {F1_Scores}
#         """)
#         text_new.close()


# # compare ill_invar with rgb
# for i in range(1,2):
#     equal_observations = True
#     nb_classes = 4

#     ##########################################################
#     img_rows, img_cols = 40, 40

#     if i == 0:
#         img_channels = 4
#         use_ill_invar = 'ill_invar'

#     elif i == 1:
#         img_channels = 3
#         use_ill_invar = 'rgb'

#     ##########################################################
#     use_gcnn = True
#     model_used = 'gcnn'
#     # print()
#     # print(model_used)
#     ##########################################################

#     for j in range(5):
#         ##########################################################
#         if j == 0:
#             train_cut = 12000 * 4
#             val_cut = 3000 * 4
#             test_cut = 3750 * 4

#         elif j == 1:
#             train_cut = 9000 * 4
#             val_cut = 2250 * 4
#             test_cut = 2812 * 4

#         elif j == 2:
#             train_cut = 6750 * 4
#             val_cut = 1687 * 4
#             test_cut = 2109 * 4

#         elif j == 3:
#             train_cut = 5062 * 4
#             val_cut = 1265 * 4
#             test_cut = 1582 * 4

#         elif j == 4:
#             train_cut = 3796 * 4
#             val_cut = 949 * 4
#             test_cut = 1186 * 4


#         # print(f'{model_used}_{use_ill_invar}_{j}')

#         cut_offs = [train_cut, val_cut, test_cut]
#         (trainX, trainY), (valX, valY), (testX, testY) = load_dataset(img_cols, img_rows, img_channels, equal_observations=equal_observations, cut_offs=cut_offs)

#         trainX = trainX.astype('float32')
#         valX = valX.astype('float32')
#         testX = testX.astype('float32')

#         Y_train = np_utils.to_categorical(trainY, nb_classes)
#         Y_val = np_utils.to_categorical(valY, nb_classes)
#         Y_test = np_utils.to_categorical(testY, nb_classes)
#         #########################################################
        

#         batch_size = 64
#         epochs = 50

#         # # Parameters for the DenseNet model builder
#         img_dim = (img_channels, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (
#             img_rows, img_cols, img_channels)

#         depth = 40
#         nb_dense_block = 3
#         growth_rate = 3  # number of z2 maps equals growth_rate * group_size, so keep this small.
#         nb_filter = 16

#         # metrics, monitor = ['acc'], 'val_acc'
#         # metrics, monitor = [f1], 'val_f1'
#         # metrics, monitor = [f1, 'acc'], 'val_f1'
#         metrics, monitor = [f1, 'acc'], 'val_acc'
#         ##########################################################

#         dropout_rate = 0.0  # 0.0 for data augmentation
#         conv_group = 'D4'  # C4 includes 90 degree rotations, D4 additionally includes reflections in x and y axis.
        

#         # Create the model (without loading weights)
#         model = GDenseNet(mc_dropout=False, padding='same', nb_dense_block=nb_dense_block, growth_rate=growth_rate,
#                         nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None, input_shape=img_dim, depth=depth,
#                         use_gcnn=use_gcnn, conv_group=conv_group, classes=nb_classes)
#         print('Model created')
#         # model.summary()
#         optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
#         model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
#         print('Finished compiling')

#         # Test equivariance by comparing outputs for rotated versions of same datapoint:
#         # res = model.predict(np.stack([trainX[123], np.rot90(trainX[123])]))
#         # is_equivariant = np.allclose(res[0], res[1])
#         # print('Equivariance check:', is_equivariant)
#         # assert is_equivariant


#         generator = ImageDataGenerator(rotation_range=15,
#                                     width_shift_range=5. / img_rows,
#                                     height_shift_range=5. / img_cols)

#         generator.fit(trainX, seed=0)

#         # ##########################################################
#         weights_file = 'model_weights.h5'
#         # ##########################################################

#         factor = 0.1
#         min_delta = 0
#         patience = nb_classes * 4

#         lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=factor,
#                                     cooldown=0, patience=nb_classes, min_lr=0.5e-6)
#         # early_stopper = EarlyStopping(monitor=f'{monitor}', min_delta=min_delta, patience=patience)
#         model_checkpoint = ModelCheckpoint(weights_file, monitor=f'{monitor}', save_best_only=True,
#                                         save_weights_only=True, mode='auto')

#         # callbacks = [lr_reducer, early_stopper, model_checkpoint]
#         callbacks = [lr_reducer, model_checkpoint]

#         model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size,
#                             epochs=epochs,
#                             callbacks=callbacks,
#                             validation_data=(valX, Y_val),
#                             verbose=1)

#         model.save(f'comparisons/gcnn_ill_invar_vs_rgb/model/{model_used}_{use_ill_invar}_{j}_model.h5')
#         print('Saved model to disk!')

#         scores = model.evaluate(testX, Y_test, batch_size=batch_size)

#         y_pred = model.predict(testX)
#         y_classes = np.argmax(y_pred, axis=-1)
#         F1_Scores = f1_score(testY, y_classes, average=None)

#         print('Test loss : ', scores[0])
#         print('Test accuracy : ', scores[1])
#         print('F1 scores : ', F1_Scores)

#         params = {'use_ill_invar': use_ill_invar, 'use_gcnn': use_gcnn, 'train_size': train_cut, 
#         'val_size': val_cut, 'test_size': test_cut, 'conv_group': conv_group}

#         text_new = open(f"comparisons/gcnn_ill_invar_vs_rgb/{model_used}_{use_ill_invar}_{j}_scores.txt", "a") 
#         n = text_new.write(f"""
#         Datashape (x_train, y_train), (x_val, y_val) (x_test, y_test) = {(trainX.shape, trainY.shape), (valX.shape, valY.shape), (testX.shape, testY.shape)}')
#         img_dimension = {img_rows, img_cols, img_channels}
#         Params : {params}
#         Monitored : {monitor}
#         Test loss : {scores[0]}
#         Test accuracy : {scores[1]}
#         F1 scores : {F1_Scores}
#         """)
#         text_new.close()
        