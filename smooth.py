import hmm
import io_utils

DIRS = [mozart, bach, beeth]
# buckets are floats with optional params ('d': dotted, 't': triplet)
BUCKETS = [[1.0/4], [1.0/4, 'd'], [1.0/8], [1.0]]

# lists of lists, 1 list per file
# H_mat[0] = [(event, note)] w/ event in (bucket, isNote)
# E_mat[0] same
# tempos is list of median tempos for reconstuction purposes
H_mat, E_mat, tempos = io_utils.generateTrainData(DIRS, BUCKETS)
model = HMM(BUCKETS)
model.train(H_mat, E_mat, laplace=0)
# inference
# generates predicted H
generated = model.predict(E_mat[0], tempos[0], BUCKETS)
io_utils.writeToMIDI(generated, BUCKETS)

