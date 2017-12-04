import midi
import math
import matplotlib.pyplot as plt
from groundtruth import NoteEvents
import glob
import copy

TEST_DIRS = ['debussy']
# needs to be floats
TEST_BUCKETS = [[1.0/16, 'd'], [1.0/16, 't'], [1.0/16],
                [1.0/8, 'd'], [1.0/8, 't'], [1.0/8],
                [1.0/4, 'd'], [1.0/4, 't'], [1.0/4],
                [1.0/2, 'd'], [1.0/2],
                [1.0]]
NOTE_TRACKS = ['Piano right']
OUTPUT_DIR = 'smoothed_midi'
ALLOWED_TEMPO_DIFF = 10

def generateTrainData(dirs, buckets, allowed_tempo_diff=ALLOWED_TEMPO_DIFF, metric=logMetric):
    assert(len(buckets) != 0)
    assert(len(dirs) != 0)
    H_mat = []
    E_mat = []
    tempos = []
    sigs = []
    filenames = getInputFiles(dirs)
    assert(len(filenames) != 0)
    # bucketfn(tick, resolution) --> bucket
    bucketfn = generateTickToBucket(buckets, metric)
    final_filenames = []
    for f in filenames:
        print f
        H, E, tempo, timeSig, keySig = processMIDI(f, bucketfn, allowed_tempo_diff)
        if tempo == None: continue
        final_filenames.append(f)
        sigs.append((timeSig, keySig))
        H_mat.append(H)
        E_mat.append(E)
        tempos.append(tempo)
    return H_mat, E_mat, tempos, final_filenames, sigs

def generateTempoErrorData(dirs, buckets, tempo_error_list, metric=squaredMetric):
    '''
    Like generateTrainData, but tempos event lists in E_mat replaced by
    list of event lists, with tempos off by percents as defined in tempo_error_list
    ex) tempo_error_list = [-10, 0, 10]
    E_mat_with_error[0] = [event list 0 with tempo off by -10%,
                           event list 0 with tempo off by 0%,
                           event list 0 with tempo off by 10%]
    '''
    print 'Gerate tempo error data'
    assert(len(buckets) != 0)
    assert(len(dirs) != 0)
    H_mat = []
    # E_mat_with_error is E_mat with each 
    E_mat_with_error = []
    # each tempo is list of tempos with errors
    tempos = []
    sigs = []
    filenames = getInputFiles(dirs)
    assert(len(filenames) != 0)
    # bucketfn(tick, resolution) --> bucket
    bucketfn = generateTickToBucket(buckets, metric)
    for f in filenames:
        H, E_with_error, tempo_list, timeSig, keySig = \
            processMIDI(f, bucketfn, allowed_tempo_diff=float('inf'), tempo_errors=tempo_error_list)
        sigs.append((timeSig, keySig))
        H_mat.append(H)
        E_mat_with_error.append(E_with_error)
        tempos.append(tempo_list)
    return H_mat, E_mat_with_error, tempos, filenames, sigs


def eventListToMIDI(eventList, buckets, ticks_per_beat, 
    tempo_bpm, filename, output_dir=OUTPUT_DIR, timeSig=None, keySig=None):
    '''
    Input: event list [(bucket, note/rest, volume)], buckets,
            ticks_per_beat, tempo (bpm)
    Writes MIDI file corresponding to event list to output_dir
    TODO: keep track of key signature/time signature info
    '''
    pattern = midi.Pattern()
    pattern.resolution = ticks_per_beat
    track = midi.Track()
    track.make_ticks_abs()
    tempoEvent = midi.events.SetTempoEvent()
    tempoEvent.set_bpm(tempo_bpm)
    track.append(tempoEvent)
    print timeSig
    print timeSig.tick
    if timeSig:
        timeSig.tick = 0
        track.append(timeSig)
    if keySig:
        keySig.tick = 0
        track.append(keySig)
    pattern.append(track)
    tick = 0
    for bucket, notes, volume in eventList:
        for note in notes:
            noteEvent = midi.NoteOnEvent(tick=int(tick), pitch=note, velocity=volume)
            track.append(noteEvent)
        tick += bucketToTicks(bucket, ticks_per_beat)
        for note in notes:
            noteEvent = midi.NoteOnEvent(tick=int(tick), pitch=note, velocity=0)
            track.append(noteEvent)
    track.append(midi.EndOfTrackEvent(tick=int(tick)))
    track.make_ticks_rel()
    print track[:10]
    midi.write_midifile(output_dir+'/'+filename, pattern)

def generateHMMMatrix(mat):
    # Generates matrix formatted for HMM processing from event list
    # Input: mat with mat[0] = [(bucket, note_list, volume), ...]
    # Output: HMM_mat with HMM_mat[0] = [(bucket, note/rest (T/F)), ...]
    HMM_mat = []
    for eventList in mat:
        HMM_list = []
        for event in eventList:
            HMM_list.append((event[0], len(event[1])!=0 ))
        HMM_mat.append(HMM_list)
    return HMM_mat

def incorporatePrediction(full_mat, pred_mat):
    # returns updated matrix that is full_mat with buckets replaced with buckets
    # in pred_mat
    # full_mat with full_mat[0] = [(bucket, note_list, volume), ...]
    # pred_mat with pred_mat[0] = [(bucket, note/rest (T/F)), ...]
    updated = copy.deepcopy(full_mat)
    for i in range(len(full_mat)):
        for j in range(len(full_mat[i])):
            updated[i][j] = (pred_mat[i][j][0], updated[i][j][1], updated[i][j][2])
    return updated


def bucketToTicks(bucket, ticks_per_beat):
    note_proportion = 0
    if len(bucket) == 1:
        note_proportion = 4 * bucket[0]
    elif bucket[1] == 't':
        note_proportion = bucket[0] * 4 * 2 / 3
    elif bucket[1] == 'd':
        note_proportion = bucket[0] * 4 * 3 / 2
    return ticks_per_beat * note_proportion




def getInputFiles(dirs):
    files = []
    for d in dirs:
        for f in glob.glob(d+'/*.mid'):
            files.append(f)
    return files

def linearMetric(t1, t2):
    return abs(t2 - t1)

def squaredMetric(t1, t2):
    return (t2 - t1) ** 2

def logMetric(t1, t2):
    t1 = float(t1)
    t2 = float(t2)
    return abs(math.log(t2) - math.log(t1))

def squaredLogMetric(t1, t2):
    t1 = float(t1)
    t2 = float(t2)
    return (math.log(t2) - math.log(t1)) ** 2

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
        # print best[0]
        # TODO: add parameter for minimum bucket
        if best[0] == (1.0/16, 't'):
            #print 'testing 16th triplet'
            # print tick, best[1]
            expected = 1.0/16 * 4 * 2 / 3 * resolution
            # print expected
            if tick < .75 * expected:
                #print 'Fake'
                return None
        return best[0]

    return bucketfn

def processMIDI(f, bucketfn, allowed_tempo_diff, tempo_errors=None):
    pattern = midi.read_midifile(file(f))
    tempoInfo = getTempoInfo(pattern)
    timeSignatureEvent = getFirstEvent(pattern, midi.events.TimeSignatureEvent)
    keySignatureEvent = getFirstEvent(pattern, midi.events.KeySignatureEvent)
    # print tempoInfo
    if not isValidTempoInfo(tempoInfo, allowed_tempo_diff):
        return [None] * 5
    tempo = tempoInfo[2]
    ticks_per_beat = pattern.resolution
    # generate E list
    # timeList [(midi_note_event, absolute time)]
    # midi_note_event ticks are absolute
    timeList = NoteEvents(pattern, note_tracks=NOTE_TRACKS, \
        start_on_note=True).note_time_list
    # print timeList
    # generate list of events with absolute ticks corresponding
    tickList_H = timeToHTickList(timeList)
    H = tickToEventList(tickList_H, bucketfn, ticks_per_beat)
    if tempo_errors == None:
        tickList_E = timeToETickList(timeList, tempo, ticks_per_beat)
        E = tickToEventList(tickList_E, bucketfn, ticks_per_beat)
        return H, E, tempo, timeSignatureEvent, keySignatureEvent
    else:
        E_with_errors = []
        tempo_list = []
        for percent_err in tempo_errors:
            curr_tempo = int(tempo * ((100.0 + percent_err) / 100))
            # print percent_err, curr_tempo
            tickList_E = timeToETickList(timeList, curr_tempo, ticks_per_beat)
            curr_E = tickToEventList(tickList_E, bucketfn, ticks_per_beat)
            E_with_errors.append(curr_E)
            tempo_list.append(curr_tempo)
        return H, E_with_errors, tempo_list, timeSignatureEvent, keySignatureEvent
    # print len(E)
    # print E[:5]
    # print len(H)
    # print H[:5]

def getFirstEvent(pattern, eventType):
    for track in pattern:
        for event in track:
            if type(event) == eventType:
                return event
    return None



def timeToETickList(timeList, tempo_bpm, ticks_per_beat):
    # Input: list [(midi_event, absolute_time)], tempo_bpm, ticks_per_beat
    # Output: list [(midi_event, absolute_tick)]
    # Will not modify midi_event tick length
    tempo = bpm_to_mpqn(tempo_bpm)
    ms_per_tick = tempo / ticks_per_beat
    print 'ms_per_tick', ms_per_tick
    tickList = [(event, ms / ms_per_tick) for event, ms in timeList]
    return tickList

def timeToHTickList(timeList):
    # Input: list [(midi_event, absolute_time)]
    # Output: list [(midi_event, original_absolute_tick)]
    # Will not modify midi_event tick length
    assert(len(timeList) != 0)
    firstNoteTick = timeList[0][0].tick
    return [(event, event.tick - firstNoteTick) for event, ms in timeList]

def tickToEventList(tickList, bucketfn, ticks_per_beat):
    # Input: list [(midi_event, tick)]
    # Output: list [(bucket, note/rest, volume)]
    # Assumes all NoteOnEvents
    # Note: may not have same length as tick list bc insert rests/may remove async notes
    result = []
    currNotes = []
    prevTick = 0
    currVolume = 0
    for event, tick in tickList:
        if type(event) == midi.events.EndOfTrackEvent:
            break
        tickDiff = tick - prevTick
        pitch, volume = event.data
        if tickDiff == 0:
            if volume != 0:
                currNotes.append(pitch)
                currVolume = volume
            else:
                currNotes = []
        else:
            bucket = bucketfn(tickDiff, ticks_per_beat)
            result.append((bucket, currNotes, currVolume))
            if volume == 0:
                currNotes = []
            else:
                currNotes = [pitch]
            currVolume = volume
            prevTick = tick
    return result


def mpqn_to_bpm(mpqn):
    return 6e7 / mpqn

def bpm_to_mpqn(bpm):
    return 6e7 / bpm

def getTempoInfo(pattern):
    # returns min, max, and quartiles of tempo
    # assumes pattern[0] has tempo events
    tempos = [event.get_mpqn() for event in pattern[0] if (type(event) == midi.events.SetTempoEvent)]
    tempos = sorted(tempos)
    ticks_per_beat = pattern.resolution
    length = len(tempos)
    results = [tempos[0], tempos[length/4], tempos[length/2], tempos[3*length/4], tempos[-1]]
    return map(mpqn_to_bpm, reversed(results))

def isValidTempoInfo(tempoInfo, allowed_tempo_diff):
    return tempoInfo[3] - tempoInfo[1] <= allowed_tempo_diff


if __name__ == '__main__':
    H_mat, E_mat_with_error, tempo_list, filenames, sigs = generateTempoErrorData(TEST_DIRS, TEST_BUCKETS, [-20, -10, 0, 10, 20])
    print len(E_mat_with_error)
    print len(E_mat_with_error[0])
    print len(E_mat_with_error[0][0])
    for i in range(len(E_mat_with_error[0])):
        print E_mat_with_error[0][i][:5]
    print tempo_list

    # H_mat, E_mat, tempos, filenames, signatures = generateTrainData(TEST_DIRS, TEST_BUCKETS)
    # print tempos, filenames
    # print len(H_mat[0])
    # print len(E_mat[0])
    # print E_mat[0][:10]
    # for i in range(len(E_mat)):
    #     timeSig, keySig = signatures[i]
    #     eventListToMIDI(E_mat[i], TEST_BUCKETS, 480, tempos[i], \
    #         filenames[i][:-4]+'_E.mid', timeSig=timeSig, keySig=keySig)
    # print bf(480, 480)
    # print bf(240, 480)
    # print bf(200, 480)
    # print bf(360, 480)
    # print bf(160, 480)
    # print bf(180, 480)






