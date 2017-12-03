from hmm import HMM
import io_utils

OUTPUT_DIR = 'smoothed_midi_3'

DIRS = ['debussy', 'mozart', 'beeth']
#DIRS = ['mozart', 'mendelssohn']
# buckets are floats with optional params ('d': dotted, 't': triplet)
BUCKETS = [(1.0/16, 'd'), (1.0/16, 't'), (1.0/16,),
            (1.0/8, 'd'), (1.0/8, 't'), (1.0/8,),
            (1.0/4, 'd'), (1.0/4, 't'), (1.0/4,),
            (1.0/2, 'd'), (1.0/2,),
            (1.0,)]

def possibleHMMBuckets(buckets):
    # returns list of ((bucket, bool)) for all possible combos
    notes = [(bucket, True) for bucket in buckets]
    rests = [(bucket, False) for bucket in buckets]
    return notes+rests

# lists of lists, 1 list per file
# H_mat[0] = [(bucket, note_list, volume)] w/ bucket in BUCKETS
# note_list list of pitches (MIDI standard, volume int
# E_mat[0] same
# tempos is list of median tempos for reconstuction purposes
# signatures [(timeSig, keySig)] midiEvents for each file
H_mat, E_mat, tempos, filenames, signatures = io_utils.generateTrainData(DIRS, BUCKETS)
model = HMM(possibleHMMBuckets(BUCKETS), nGramLength=2, laplace=20)
HMM_H = io_utils.generateHMMMatrix(H_mat)
HMM_E = io_utils.generateHMMMatrix(E_mat)
# print HMM_H[0][:10]
model.train(HMM_H, HMM_E)
# print '====================='
# for key, prob in model.transProbs.iteritems():
#     max_bucket = max(prob, key=prob.get)
#     print key , ':', max_bucket, prob[max_bucket]
predictions = model.predict_bigram(HMM_E)

smoothed_E = io_utils.incorporatePrediction(E_mat, predictions)
for i in range(len(smoothed_E)):
    timeSig, keySig = signatures[i]
    io_utils.eventListToMIDI(smoothed_E[i], BUCKETS, 480, tempos[i], \
            filenames[i][:-4]+'_E_smoothed.mid', output_dir=OUTPUT_DIR, \
            timeSig=timeSig, keySig=keySig)
    
    io_utils.eventListToMIDI(E_mat[i], BUCKETS, 480, tempos[i], \
        filenames[i][:-4]+'_E_orig.mid', output_dir=OUTPUT_DIR, \
        timeSig=timeSig, keySig=keySig)



# for key, prob in model.transProbs.iteritems():
#     max_bucket = max(prob, key=prob.get)
#     print key , ':', max_bucket, prob[max_bucket]
# print '====================='
# for key, prob in model.emissionProbs.iteritems():
#     print key, ':', max(prob)
# inference
# generates predicted H
# generated = model.predict(E_mat[0], tempos[0], BUCKETS)
# io_utils.writeToMIDI(generated, BUCKETS)

