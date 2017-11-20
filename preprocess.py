import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

DOWNSAMPLED_SR = 16000
HOP_LENGTH = 512
NUM_BINS = 252
BINS_PER_OCTAVE = 36
NAME = "chpn_a_flat"

def preprocess(filename, offset=0,duration=None):
	y, sr = librosa.load(filename, sr=None, offset=offset, duration=duration)
	print sr
	y_downsample = librosa.resample(y, orig_sr=sr, target_sr=DOWNSAMPLED_SR)
	print y_downsample.shape
	C = librosa.cqt(y_downsample, sr=DOWNSAMPLED_SR, hop_length=HOP_LENGTH, n_bins=NUM_BINS, bins_per_octave=BINS_PER_OCTAVE)
	mean = np.mean(C, axis=1, keepdims=True)
	std = np.std(C, axis=1, keepdims=True)
	C = np.divide(np.subtract(C, mean), std)
	return C

def plotCBT(C,output):
	librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),sr=DOWNSAMPLED_SR, x_axis='time', y_axis='cqt_note')
	plt.colorbar(format='%+2.0f dB')
	plt.title('Constant-Q power spectrum Resampled')
	plt.tight_layout()
	plt.savefig(output)

def main():
	filename = NAME + ".wav"
	#duration = librosa.get_duration(filename=filename)
	#offset = duration - 5
	C = preprocess(filename,duration=10)
	amp = librosa.amplitude_to_db(C)
	print C.shape
	"""plotCBT(C,"CQT_Resampled_FirstSecond_{}.png".format(NAME))"""

if __name__ == "__main__":
	main()