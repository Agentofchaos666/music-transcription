import numpy as np
from keras.models import Model
from keras.models import load_model
from tensorflow.python.lib.io import file_io
import argparse
from createInputs import get_wav_midi_data
import io_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-prefix',help='Filename to predict as MIDI',required=True)
    parser.add_argument('--model-path',help='Path for model .h5 file',required=True)
    args = parser.parse_args()
    arguments = args.__dict__

    prefix = arguments['file_prefix']
    model_path = arguments['model_path']

    model = load_model(model_path)
    X, Y = get_wav_midi_data([(prefix + '.wav', prefix + '.mid')])
    prediction = np.squeeze(np.asarray(model.predict(X))).round() # of shape (88, numSlices)
    predictionToBasicMIDI(y_pred, prefix)

if __name__ == "__main__":
    main()
