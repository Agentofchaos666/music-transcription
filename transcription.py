import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from groundtruth import NoteEvents
import midi
import glob
import time

from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, AveragePooling2D, BatchNormalization, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import Model
from keras import layers
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint

from tensorflow.python.lib.io import file_io
import argparse

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
BATCH_SIZE = 64
TRAINING_DIRS = [] 
BUCKET_NAME = "output/" #CHANGE TO BUCKET NAME

def plot_prediction(prediction, target):
    prediction = np.squeeze(prediction) # print prediction.shape
    target = [np.squeeze(arr) for arr in target] # print len(target), target[0].shape
    plt.matshow(prediction)
    plt.savefig('prediction.png')
    plt.clf()
    plt.matshow(target)
    plt.savefig('target.png')

class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.losses =[]

    def on_batch_end(self, batch, logs={}):
        # print '=====> PRINTING LOGS FOR BATCH [', batch, '] ....'
        # for key, val in sorted(logs.items()):
        #     print key, ':', val
        # print '=====> FINISHED PRINTING LOGS.'
        self.losses.append(logs.get('loss'))

def storeCloud(filename):
    with file_io.FileIO(filename, mode='r') as input_f:
        with file_io.FileIO(BUCKET_NAME + filename, mode='w+') as output_f:
                output_f.write(input_f.read())

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.train_begin_time = time.time() - self.model.fit_start_time
        print 'Time since fit() was called:', self.train_begin_time
        print '===> BEGINNING TO TRAIN...'
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions =[]
        self.error_file = "error_output.txt"
        self.model_file = "model.h5"
        self.target_file = "target.npy"
        self.predict_file = "predict.npy"
        with open(self.error_file, 'w') as error_output:
            error_output.write("")   #reset file

    def on_epoch_begin(self, epoch, logs={}):
        if epoch == 0:
            time_since_training_began = time.time() - self.train_begin_time
            print 'Time since training began:', time_since_training_began

        print 'EPOCH [', epoch, ']:'
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        end_time = time.time() - self.start_time
        print 'Train:', end_time
        acc_scores = []
        f1_scores = []
        recall_scores = []
        precision_scores =[]

        val_predict = np.asarray(self.model.predict(self.model.validation_data[0]))
        val_predict = val_predict.round()
        val_target = self.model.validation_data[1]
        # print 'PREDICT_SHAPE:', val_predict.shape, '| TARGET_SHAPE:', len(val_target), val_target[0].shape
        if epoch == 0:
            np.save(self.target_file, val_target)
            storeCloud(self.target_file)

        elif epoch % 2 == 0:
            self.model.save(self.model_file)
            storeCloud(self.model_file)
            np.save(self.predict_file, val_predict)
            storeCloud(self.predict_file)

        error_output = open(self.error_file, 'a')
        error_output.write('----EPOCH {}----\n'.format(epoch))
        plot_prediction(val_predict[:, :626], [x[:626] for x in val_target])
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
            self.val_recalls.append(val_recall)
            self.val_precisions.append(val_precision)
            print '== NOTE {}: VAL_F1: {} | VAL_PRECISION: {} | VAL_RECALL {}\n'.format(i, val_f1, val_precision, val_recall)
            error_output.write('== NOTE {}: VAL_F1: {} | VAL_PRECISION: {} | VAL_RECALL {}\n'.format(i, val_f1, val_precision, val_recall))

        self.val_f1s.extend(f1_scores)
        inference_time = time.time() - end_time
        f1 = sum(f1_scores) / float(len(f1_scores))
        recall = sum(recall_scores) / float(len(recall_scores))
        precision = sum(precision_scores) / float(len(precision_scores))
        acc = sum(acc_scores) / float(len(acc_scores))

        print 'F1 SCORE =', f1, 
        print '| RECALL =', recall,
        print '| PRECISION =', precision,
        print '| ACC =', acc
        error_output.write('F1 SCORE = {} | RECALL = {} | PRECISION = {} |  ACCURACY = {}\n'.format(f1, recall, precision, acc))
        error_output.close()
        storeCloud(self.error_file)
            # print '== NOTE {}: F1: {} | Precision: {} | Recall: {}'.format(i, val_f1, val_precision, val_recall)
        return

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
    print '======== Y_TRUE ========='
    print y_true.data
    print '======== Y_PRED ========='
    print y_pred.data
    

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
    args = parser.parse_args()
    arguments = args.__dict__
    print arguments
    
    print '===> Setting up data...'
    data_set_up_start_time = time.time()                                                                        
    x_stream = file_io.FileIO(arguments['X_file'], mode='r')
    y_stream = file_io.FileIO(arguments['Y_file'], mode='r')
    X = np.load(x_stream)
    Y = np.load(y_stream)
    Y = [Y[i] for i in range(Y.shape[0])]
    print '===> Finished setting up data:', time.time() - data_set_up_start_time 

    model = ModelBuilder(input_shape=(252, 7, 1), 
                         num_filters=[50, 50], 
                         kernel_size_tuples=[(25,5), (5,3)], 
                         pool_size=(3,1),
                         num_hidden_units=[200, 200],
                         dropout_rate=0.1)

    lossHistory = LossHistory()
    metrics = Metrics()
    sgd = SGD(lr=LEARNING_RATE, momentum=MOMENTUM_RATE)
    print '===> Compiling the model...'
    compile_model_start_time = time.time()
    model.compile(optimizer=sgd, loss='hinge', metrics=['accuracy'])
    print '===> Finished compiling the model:', time.time() - compile_model_start_time

    model.validation_data = (X, Y)

    # EXPERIMENTING WITH PLOTTING
    val_predict = np.asarray(model.predict(X)).round()
    val_target = Y
    plot_prediction(val_predict[:, :626], [x[:626] for x in val_target])
    # plot_prediction(val_predict[:, :626], [x[:626] for x in val_target])
    model.fit_start_time = time.time()
    model.fit(X, Y, validation_data=(X, Y), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[lossHistory, metrics])


if __name__ == "__main__":
    main()
