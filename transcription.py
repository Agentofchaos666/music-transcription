import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from groundtruth import NoteEvents
import midi
import glob

from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, AveragePooling2D, BatchNormalization, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import Model
from keras import layers
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import Callback

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
TRAINING_DIRS = ['mozart'] 

def getFileList():
    file_list = []
    for train_dir in TRAINING_DIRS:
        midi_files = set([f[:-4] for f in glob.glob(train_dir+'/*.mid')])
        wav_files = set([f[:-4] for f in glob.glob(train_dir+'/*.wav')])
        train_files = midi_files & wav_files
        # print train_files
        for filename in train_files:
            file_list.append((filename+'.wav',filename+'.mid'))
    return file_list

def preprocess_wav_file(files, Y_numSlices):
    # returns 1 example (downsampled, cqt, normalized)
    np_array_list = []
    for filename in files:
        y, sr = librosa.load(filename, sr = None, duration=DURATION)
        y_downsample = librosa.resample(y, orig_sr=sr, target_sr=DOWNSAMPLED_SR)
        CQT_result = librosa.cqt(y_downsample, sr=DOWNSAMPLED_SR, hop_length=HOP_LENGTH, n_bins=NUM_BINS, bins_per_octave=BINS_PER_OCTAVE)
        CQT_result = np.absolute(CQT_result)
        np_array_list.append(CQT_result)

    combined = np.concatenate(np_array_list, axis = 1)
    mean = np.mean(combined, axis = 1, keepdims =True)
    std = np.std(combined, axis = 1, keepdims=True)
    for i in range(len(np_array_list)):
        np_array_list[i] = np.divide(np.subtract(np_array_list[i], mean), std)

    frame_windows_list = []
    numSlices_list = []
    for i in range(len(np_array_list)):
        CQT_result = np_array_list[i]
        paddedX = np.zeros((CQT_result.shape[0], CQT_result.shape[1] + WINDOW_SIZE - 1), dtype=float)
        pad_amount = WINDOW_SIZE / 2
        paddedX[:, pad_amount:-pad_amount] = CQT_result
        frame_windows = np.array([paddedX[:, j:j+WINDOW_SIZE] for j in range(CQT_result.shape[1])])
        frame_windows = np.expand_dims(frame_windows, axis=3)
        numSlices = min(frame_windows.shape[0],Y_numSlices[i])
        numSlices_list.append(numSlices)
        frame_windows_list.append(frame_windows[:numSlices])
    return np.concatenate(frame_windows_list, axis=0), numSlices_list

def preprocess_midi_truth(filename):
    # returns 1 ground truth binary vector (size 88)
    pattern = midi.read_midifile(file(filename))
    events = NoteEvents(pattern)
    truth = events.get_ground_truth(31.25, DURATION) # (88, numSlices)
    return truth

def get_wav_midi_data(filenames):
    X_filenames = []
    Y_numSlices = []
    Y_list = []
    for wav_file, midi_file in filenames:
        X_filenames.append(wav_file)
        Y_i = preprocess_midi_truth(midi_file)
        Y_numSlices.append(Y_i.shape[1])
        Y_list.append(Y_i)

    X, numSlices = preprocess_wav_file(X_filenames, Y_numSlices)
    Y_list = [Y_list[i][:,:numSlices[i]] for i in range(len(Y_list))]
    Y = np.concatenate(Y_list, axis=1)
    Y = [Y[i] for i in range(Y.shape[0])]
    return X, Y

def plot_prediction(prediction, target,name):
    prediction = np.squeeze(prediction)
    target = [np.squeeze(arr) for arr in target]
    print prediction.shape
    print len(target), target[0].shape
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

        # val_predict = np.asarray(self.model.predict(self.model.validation_data[0]))
        # val_predict = val_predict.round()
        # val_target = self.model.validation_data[1]
        # plot_prediction(val_predict[:, :626], [x[:626] for x in val_target])

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions =[]

    def on_epoch_begin(self, epoch, logs={}):
        print 'EPOCH [', epoch, ']:',

    def on_epoch_end(self, epoch, logs={}):
        acc_scores = []
        f1_scores = []
        recall_scores = []
        precision_scores =[]

        val_predict = np.asarray(self.model.predict(self.model.validation_data[0]))
        val_predict = val_predict.round()
        val_target = self.model.validation_data[1]
        # print 'PREDICT_SHAPE:', val_predict.shape, '| TARGET_SHAPE:', len(val_target), val_target[0].shape
        plot_prediction(val_predict[:, :626], [x[:626] for x in val_target])
        for i in range(val_predict.shape[0]):
            # pred = np.random.uniform(size=(3750,1)).round() 
            # target = np.random.uniform(size=(3750,1)).round()
            pred = val_predict[i] 
            target = val_target[i] 
            # print 'predddddddDDD:', pred
            # print '===================================='
            # print 'TARRRRRGETTT:', target
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
            print '== NOTE {}: VAL_F1: {} | VAL_PRECISION: {} | VAL_RECALL {}'.format(i, val_f1, val_precision, val_recall)
        self.val_f1s.extend(f1_scores)
        print 'F1 SCORE =', sum(f1_scores) / float(len(f1_scores)), 
        print '| RECALL =', sum(recall_scores) / float(len(recall_scores)),
        print '| PRECISION =', sum(precision_scores) / float(len(precision_scores)),
        print '| ACC =', sum(acc_scores) / float(len(acc_scores))
        return

def ModelBuilder(input_shape, num_filters, kernel_size_tuples, pool_size, num_hidden_units, dropout_rate):
    frame_input = Input(shape=input_shape)
    x = Convolution2D(filters=num_filters[0], kernel_size=kernel_size_tuples[0], padding='same', kernel_initializer='he_normal')(frame_input)
    x = Activation('tanh')(x)
    x = MaxPooling2D(pool_size)(x)
    # x = Dropout(dropout_rate)(x)

    x = Convolution2D(filters=num_filters[1], kernel_size=kernel_size_tuples[1], padding='same', kernel_initializer='he_normal')(x)
    x = Activation('tanh')(x)
    x = MaxPooling2D(pool_size)(x)    
    # x = Dropout(dropout_rate)(x)

    x = Flatten()(x)
    x = Dense(num_hidden_units[0], activation='sigmoid')(x)
    # x = Dropout(dropout_rate)(x)
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
    model = ModelBuilder(input_shape=(252, 7, 1), 
                         num_filters=[50, 50], 
                         kernel_size_tuples=[(25,5), (5,3)], 
                         pool_size=(3,1),
                         num_hidden_units=[200, 200],
                         dropout_rate=0.1)
    # model.summary()
    print '===> Setting up data...'
    X, Y = get_wav_midi_data(getFileList())
    print '===> Finished setting up data.'
    print '========================================'
    lossHistory = LossHistory()
    metrics = Metrics()
    sgd = SGD(lr=LEARNING_RATE, momentum=MOMENTUM_RATE)
    print '===> Compiling the model...'
    model.compile(optimizer=sgd, loss='hinge', metrics=['accuracy'])
    print '===> Finished compiling the model.'
    print '========================================'
    model.validation_data = (X, Y)

    # EXPERIMENTING WITH PLOTTING
    val_predict = np.asarray(model.predict(X)).round()
    val_target = Y
    plot_prediction(val_predict[:, :626], [x[:626] for x in val_target])

    model.fit(X, Y, validation_data=(X, Y), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[lossHistory, metrics])


if __name__ == "__main__":
    main()
