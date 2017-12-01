import midi
import math
import matplotlib.pyplot as plt
from groundtruth import NoteEvents
import glob

TEST_DIRS = ['mozart']
# needs to be floats
TEST_BUCKETS = [[1.0/4], [1.0/8, 'd'], [1.0/8, 't'], [1.0/8]]
NOTE_TRACKS = ['Piano right']
ALLOWED_TEMPO_DIFF = 10

def generateTrainData(dirs, buckets):
    assert(len(buckets) != 0)
    assert(len(dirs) != 0)
    H_mat = []
    E_mat = []
    tempos = []
    files = getInputFiles(dirs)
    assert(len(files) != 0)
    # bucketfn(tick, resolution) --> bucket
    bucketfn = generateTickToBucket(buckets, linearMetric)
    for f in files:
        print f
        H, E, tempo = processMIDI(f, bucketfn)
        break
        if tempo == None: continue
        H_mat.append(H)
        E_mat.append(E)
        tempos.append(tempo)
    return H_mat, E_mat, tempos

def getInputFiles(dirs):
    files = []
    for d in dirs:
        for f in glob.glob(d+'/*.mid'):
            files.append(f)
    return files

def linearMetric(t1, t2):
    return abs(t2 - t1)

def logMetric(t1, t2):
    t1 = float(t1)
    t2 = float(t2)
    return abs(math.log(t2) - math.log(t1))

def generateTickToBucket(buckets, metric):
    # proportion of resolution actually corresponding to each bucket
    # [1/4] --> 1/4 * 4 bc 1 quarter note = resolution ticks
    # [1/4, t] --> 1/4 * 4 * 2/3 bc triplet is 2/3 length
    # [1/4, d] --> 1/4 * 4 * 3/2 bc dotted note is 3/2 length
    note_proportions = []
    for bucket in buckets:
        # print bucket
        # print len(bucket)
        if len(bucket) == 1:
            note_proportions.append(bucket[0] * 4)
        elif bucket[1] == 't':
            note_proportions.append(bucket[0] * 4 * 2 / 3)
        elif bucket[1] == 'd':
            note_proportions.append(bucket[0] * 4 * 3 / 2)
    print note_proportions
	
    def bucketfn(tick, resolution):
        best = (buckets[0], metric(tick, resolution * note_proportions[0]))
        for index, note_prop in enumerate(note_proportions):
            # print 'index/np: ', index, note_prop
            # print note_prop * resolution
            dist = metric(tick, resolution * note_prop)
            # print dist
            if dist < best[1]:
                # print 'update'
                best = (buckets[index], dist)
                # print best
        return best[0]

    return bucketfn

def processMIDI(f, bucketfn):
    pattern = midi.read_midifile(file(f))
    tempoInfo = getTempoInfo(pattern)
    print tempoInfo
    if not isValidTempoInfo(tempoInfo):
        return None, None, None
    tempo = tempoInfo[2]
    # generate E list
    # timeList [(midi_note_event, absolute time)]
    # midi_note_event ticks are absolute
    timeList = NoteEvents(pattern, note_tracks=NOTE_TRACKS, \
        start_on_note=True)._note_time_list
    print timeList
    # generate list of events with absolute ticks corresponding
    # to absolute time with given tempo
    # process list to handle simultaneous notes/have rests/
    # be properly consecutive
    # convert to E list

    # H list generation
    # generate note event - tick list from groundtruth
    # process list to handle simultaneous notes/have rests/
    # be properly consecutive
    # convert to H list

def mpqn_to_bpm(mpqn):
    return 6e7 / mpqn

def getTempoInfo(pattern):
    # returns min, max, and quartiles of tempo
    # assumes pattern[0] has tempo events
    tempos = [event.get_mpqn() for event in pattern[0] if (type(event) == midi.events.SetTempoEvent)]
    tempos = sorted(tempos)
    ticks_per_beat = pattern.resolution
    length = len(tempos)
    results = [tempos[0], tempos[length/4], tempos[length/2], tempos[3*length/4], tempos[-1]]
    return map(mpqn_to_bpm, reversed(results))

def isValidTempoInfo(tempoInfo):
    return tempoInfo[3] - tempoInfo[1] <= ALLOWED_TEMPO_DIFF


if __name__ == '__main__':
    generateTrainData(TEST_DIRS, TEST_BUCKETS)
    # print bf(480, 480)
    # print bf(240, 480)
    # print bf(200, 480)
    # print bf(360, 480)
    # print bf(160, 480)
    # print bf(180, 480)






