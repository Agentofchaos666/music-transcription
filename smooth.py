import hmm
import io_utils

DIRS = [mozart, bach, beeth]
BUCKETS = ['1/4', '1/4d', '1/8', '1']

# lists of lists, 1 list per file
# H_mat[0] = [(event, note)] w/ event in (bucket, isNote)
# E_mat[0] same
# tempos is list of median tempos for reconstuction purposes
H_mat, E_mat, tempos = io_utils.generateTrainData(DIRS, BUCKETS)
model = HMM()
model.train(H_mat, E_mat)
# inference
# generates predicted H
generated = model.predict(E_mat[0], tempos[0], BUCKETS)
io_utils.writeToMIDI(generated, BUCKETS)

