import midi
import math
import matplotlib.pyplot as plt
from groundtruth import NoteEvents
import glob

TEST_DIRS = ['debussy']
# needs to be floats
TEST_BUCKETS = [[1.0/16, 'd'], [1.0/16, 't'], [1.0/16],
                [1.0/8, 'd'], [1.0/8, 't'], [1.0/8],
                [1.0/4, 'd'], [1.0/4, 't'], [1.0/4],
                [1.0/2, 'd'], [1.0/2],
                [1.0]]
NOTE_TRACKS = ['Piano right']
OUTPUT_DIR = 'smoothed_midi'
ALLOWED_TEMPO_DIFF = 100

def generateTrainData(dirs, buckets):
    assert(len(buckets) != 0)
    assert(len(dirs) != 0)
    H_mat = []
    E_mat = []
    tempos = []
    sigs = []
    filenames = getInputFiles(dirs)
    assert(len(filenames) != 0)
    # bucketfn(tick, resolution) --> bucket
    bucketfn = generateTickToBucket(buckets, linearMetric)
    for f in filenames:
        print f
        H, E, tempo, timeSig, keySig = processMIDI(f, bucketfn)
        sigs.append((timeSig, keySig))
        if tempo == None: continue
        H_mat.append(H)
        E_mat.append(E)
        tempos.append(tempo)
    return H_mat, E_mat, tempos, filenames, sigs

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
    timeSignatureEvent = getFirstEvent(pattern, midi.events.TimeSignatureEvent)
    keySignatureEvent = getFirstEvent(pattern, midi.events.KeySignatureEvent)
    # print tempoInfo
    if not isValidTempoInfo(tempoInfo):
        return (None, None, None)
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
    tickList_E = timeToETickList(timeList, tempo, ticks_per_beat)
    # print timeList[:10]
    # print tickList_E[:10]
    # print tickList_H[:10]
    E = tickToEventList(tickList_E, bucketfn, ticks_per_beat)
    H = tickToEventList(tickList_H, bucketfn, ticks_per_beat)
    # print len(E)
    # print E[:5]
    # print len(H)
    # print H[:5]
    return H, E, tempo, timeSignatureEvent, keySignatureEvent
    # to absolute time with given tempo
    # process list to handle simultaneous notes/have rests/
    # be properly consecutive
    # convert to E list

    # H list generation
    # generate note event - tick list from groundtruth
    # process list to handle simultaneous notes/have rests/
    # be properly consecutive
    # convert to H list

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

def isValidTempoInfo(tempoInfo):
    return tempoInfo[3] - tempoInfo[1] <= ALLOWED_TEMPO_DIFF


if __name__ == '__main__':
    H_mat, E_mat, tempos, filenames, signatures = generateTrainData(TEST_DIRS, TEST_BUCKETS)
    print tempos, filenames
    print len(H_mat[0])
    print len(E_mat[0])
    print E_mat[0][:10]
    for i in range(len(E_mat)):
        timeSig, keySig = signatures[i]
        eventListToMIDI(E_mat[i], TEST_BUCKETS, 480, tempos[i], \
            filenames[i][:-4]+'_E.mid', timeSig=timeSig, keySig=keySig)
    # print bf(480, 480)
    # print bf(240, 480)
    # print bf(200, 480)
    # print bf(360, 480)
    # print bf(160, 480)
    # print bf(180, 480)






