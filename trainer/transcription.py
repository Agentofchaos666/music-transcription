import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import time
from StringIO import StringIO

from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, AveragePooling2D, BatchNormalization, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import Model
from keras import layers
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import Callback
from keras.callbacks import LearningRateScheduler
from keras.models import load_model

from tensorflow.python.lib.io import file_io
import argparse
import h5py

############################
import tensorflow as tf 
from keras.backend.tensorflow_backend import set_session 
set_session(tf.Session())
############################

DURATION = 60
DOWNSAMPLED_SR = 16000
HOP_LENGTH = 512
NUM_OCTAVES = 7
BINS_PER_OCTAVE = 36
NUM_BINS = NUM_OCTAVES * BINS_PER_OCTAVE

WINDOW_SIZE = 7
LEARNING_RATE = 0.1
MOMENTUM_RATE = 0.9
NUM_EPOCHS = 100
BATCH_SIZE = 128
TRAINING_DIRS = [] 

def plot_predictions(plotting_info):
    for prediction, target, epoch, i in plotting_info:
        suffix = '_' + str(epoch) + '_' + str(i) + '.png'
        prediction = np.squeeze(prediction) # print prediction.shape
        target = [np.squeeze(arr) for arr in target] # print len(target), target[0].shape
        plt.subplot(2,1,1)
        plt.matshow(prediction, fignum=False)
        plt.subplot(2,1,2)
        plt.matshow(target, fignum=False)
        plt.savefig('comparison-plots/comparison' + suffix)
        plt.clf()

class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.losses =[]

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.train_begin_time = time.time()
        print 'Time since fit() was called:', self.train_begin_time - self.model.after_compile_start_time
        print '===> BEGINNING TO TRAIN...'
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions =[]

    def on_epoch_begin(self, epoch, logs={}):
        print 'EPOCH [', epoch, ']:'
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        end_time = time.time()
        print '---> Train:', end_time - self.start_time 
        
        if epoch > 2:
            data_to_use = [self.model.train_data, self.model.validation_data]
        else: 
            data_to_use = [self.model.validation_data]
        for X, Y in data_to_use:
            evaluate_model(self.model, X, Y)
            inference_time = time.time()
            print '---> Inference Time for above:', inference_time - end_time
            end_time = inference_time

        print '*********************** max f1 score =', max(f1_scores), '| idx =', f1_scores.index(max(f1_scores))
        max_idx_to_plot = val_target[0].shape[0]-627
        indices_to_visualize = [max_idx_to_plot, min(f1_scores.index(max(f1_scores)), max_idx_to_plot)]
        plot_predictions([(val_predict[:, i:i+626], [x[i:i+626] for x in val_target], epoch, i) for i in indices_to_visualize])


class Checkpoint(Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save("model.h5")

def step_decay_3(epoch):
    initial_lrate = 0.05
    if epoch < 5:
        lrate = initial_lrate
    elif epoch < 12:
        lrate = initial_lrate / 2
    elif epoch < 20: 
        lrate = initial_lrate / 4
    else:
        lrate = initial_lrate / 8
    return lrate

def ModelBuilder(input_shape, num_filters, kernel_size_tuples, pool_size, num_hidden_units, dropout_rate):
    frame_input = Input(shape=input_shape)
    x = Convolution2D(filters=num_filters[0], kernel_size=kernel_size_tuples[0], padding='same', kernel_initializer='he_normal')(frame_input)
    x = Activation('tanh')(x)
    x = MaxPooling2D(pool_size)(x)
    x = Dropout(dropout_rate)(x)

    x = Convolution2D(filters=num_filters[1], kernel_size=kernel_size_tuples[1], padding='same', kernel_initializer='he_normal')(x)
    x = Activation('tanh')(x)
    x = MaxPooling2D(pool_size)(x)    
    x = Dropout(dropout_rate)(x)

    x = Flatten()(x)
    x = Dense(num_hidden_units[0], activation='sigmoid')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_hidden_units[1], activation='sigmoid')(x)
    x = Dropout(dropout_rate)(x)
    outputs = []
    name_base = 'note_'
    for i in range(88):
        name = name_base + str(i)
        outputs.append(Dense(1, activation='sigmoid', name=name)(x))
        
    return Model(inputs=frame_input, outputs=outputs) 

def custom_loss_function(y_true, y_pred):
    # K.print_tensor(K.shape(y_true))

    # ones = K.ones(shape=K.int_shape(y_true))
    # weights = K.update_add(ones, y_true)
    binary_cross_entropy = K.binary_crossentropy(y_true, y_pred)
    return K.mean(binary_cross_entropy * K.exp(y_true))
    # K.print_tensor(weights, 'WEIGHTS: ')


    print '======== Y_TRUE ========='
    print type(y_true)
    # y_true_array = tf.Session().run(y_true)
    # print type(y_true_array), y_true_array.shape, y_true_array[0]
    print '======== Y_PRED ========='
    print type(y_pred)
    # y_pred_array = tf.Session().run(y_pred)
    # print type(y_pred_array), y_pred_array.shape, y_pred_array[0]
 
def evaluate_model(model, X, Y):   
    acc_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores =[]

    val_predict = np.asarray(model.predict(X))
    val_predict = val_predict.round()
    val_target = Y

    for i in range(val_predict.shape[0]):
        pred = val_predict[i] 
        target = val_target[i] 
        
        val_acc = accuracy_score(target, pred)
        val_f1 = f1_score(target, pred)
        val_recall = recall_score(target, pred)
        val_precision = precision_score(target, pred)
        
        acc_scores.append(val_acc)
        f1_scores.append(val_f1)
        recall_scores.append(val_recall)
        precision_scores.append(val_precision)

    print '---> F1:', sum(f1_scores) / float(len(f1_scores)), 
    print '| Recall:', sum(recall_scores) / float(len(recall_scores)),
    print '| Precision:', sum(precision_scores) / float(len(precision_scores)),
    print '| Acc:', sum(acc_scores) / float(len(acc_scores))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--X-file',
      help='GCS or local paths to training data',
      required=True
    )
    parser.add_argument(
      '--Y-file',
      help='GCS or local paths to training data',
      required=True
    )
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=False
    )                                                                                                   
    args = parser.parse_args()
    arguments = args.__dict__
    print arguments
    
    print '===> Setting up data...'
    data_set_up_start_time = time.time()                                                                        
    x_stream = StringIO(file_io.read_file_to_string(arguments['X_file']))
    y_stream = StringIO(file_io.read_file_to_string(arguments['Y_file']))
    X = np.load(x_stream)
    Y = np.load(y_stream)
    numSlicesForTrain = int(X.shape[0] * 0.8)
    
    X_train = X[:numSlicesForTrain] # X_train.shape = (numSlices, 252, 7, 1)
    Y_train = Y[:, :numSlicesForTrain]
    Y_train = [Y_train[i] for i in range(Y_train.shape[0])] # list of 88 things of shape (numSliceForTrain,)
    print '===> TRAIN DIMENSIONS:', X_train.shape, '|', len(Y_train), Y_train[0].shape

    X_test = X[numSlicesForTrain:] # X_test.shape = (numSlices, 252, 7, 1)
    Y_test = Y[:, numSlicesForTrain:]
    Y_test = [Y_test[i] for i in range(Y_test.shape[0])] # list of 88 things of shape (numSliceForTrain,)
    print '===> TEST DIMENSIONS:', X_test.shape, '|', len(Y_test), Y_test[0].shape

    print '===> Finished setting up data:', time.time() - data_set_up_start_time 

    # model = ModelBuilder(input_shape=(252, 7, 1), 
    #                      num_filters=[50, 50], 
    #                      kernel_size_tuples=[(25,5), (5,3)], 
    #                      pool_size=(3,1),
    #                      num_hidden_units=[1000, 200],
    #                      dropout_rate=0.3)

    # lossHistory = LossHistory()
    # metrics = Metrics()
    # sgd = SGD(lr=0.0, momentum=MOMENTUM_RATE)
    # print '===> Compiling the model...'
    # compile_model_start_time = time.time()
    # model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    # print '===> Finished compiling the model:', time.time() - compile_model_start_time

    # model.train_data = (X_train, Y_train)
    # model.validation_data = (X_test, Y_test)
    # lrate = LearningRateScheduler(step_decay_3)
    # model.after_compile_start_time = time.time()
    # model.fit(X_train, Y_train, validation_data=model.validation_data, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[lossHistory, metrics, lrate])

    print '===> Loading model...'
    model = load_model('model_shuffled.h5')
    print '===> Evaluating on train data...'
    evaluate_model(model, X_train, Y_train)

if __name__ == "__main__":
    main()
