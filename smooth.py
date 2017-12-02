import hmm
import io_utils

DIRS = [mozart, bach, beeth]
# buckets are floats with optional params ('d': dotted, 't': triplet)
BUCKETS = [[1.0/16, 'd'], [1.0/16, 't'], [1.0/16],
            [1.0/8, 'd'], [1.0/8, 't'], [1.0/8],
            [1.0/4, 'd'], [1.0/4, 't'], [1.0/4],
            [1.0/2, 'd'], [1.0/2],
            [1.0]]


def generateHMMMatrix(mat):
    HMM_mat = []
    for eventList in mat:
        HMM_list = []
        for event in eventList:
            HMM_list.append((event[0], len(event[1]==0) ))
        HMM_mat.append(HMM_list)
    return HMM_mat

def incorporatePrediction(full_mat, pred_mat):
    for i in range(len(full_mat)):
        for j in range(len(full_mat[0])):
            full_mat[i][j][0] = pred_mat[i][j][0]

# lists of lists, 1 list per file
# H_mat[0] = [(bucket, note_list, volume)] w/ bucket in BUCKETS
# note_list list of pitches (MIDI standard, volume int
# E_mat[0] same
# tempos is list of median tempos for reconstuction purposes
# signatures [(timeSig, keySig)] midiEvents for each file
H_mat, E_mat, tempos, filenames, signatures = io_utils.generateTrainData(DIRS, BUCKETS)
model = HMM(len(BUCKETS)*2)
HMM_H = generateHMMMatrix(H_mat)
HMM_E = generateHMMMatrix(E_mat)
model.train(HMM_H, HMM_E, laplace=0)
# inference
# generates predicted H
# generated = model.predict(E_mat[0], tempos[0], BUCKETS)
# io_utils.writeToMIDI(generated, BUCKETS)

