import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import midi
from groundtruth import NoteEvents
from preprocess import preprocess

DOWNSAMPLED_SR = 16000
NAME = "deb_clai"

def main():
	filename = "{}.wav".format(NAME)
	C = preprocess(filename,duration = 20)
	plt.subplot(2,1,1)
	librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),sr=DOWNSAMPLED_SR, x_axis = "time", y_axis='cqt_note')
	plt.tight_layout()
	pattern = midi.read_midifile(file('{}.mid'.format(NAME)))
	events = NoteEvents(pattern)
	truth = events.get_ground_truth(31.25)
	truth_slice = truth[:,0:626]
	truth_slice_inverted = truth[::-1,0:626]
	print truth_slice[:,0]
	print truth_slice_inverted[:,0]
	plt.subplot(2,1,2)
	plt.imshow(truth_slice_inverted,extent=[0,626,200,0])
	plt.xticks([i for i in range(0,626,100)],[i/31.25 for i in range(0,626,100)])
	plt.xlabel("Time")
	plt.yticks([])
	plt.tight_layout()
	plt.savefig("{}_comparison.png".format(NAME))


if __name__ == "__main__":
	main()