import midi
import io_utils
import numpy as np
from groundtruth import NoteEvents

SLICES_PER_SECOND = 31.25
RESOLUTION = SLICES_PER_SECOND * 12

TEST_BUCKETS = [(1.0/64,), (1.0/32,), (1.0/16, 'd'), (1.0/16, 't'), (1.0/16,),
                (1.0/8, 'd'), (1.0/8, 't'), (1.0/8,),
                (1.0/4, 'd'), (1.0/4, 't'), (1.0/4,),
                (1.0/2, 'd'), (1.0/2,),
                (1.0,), (2.0,), (3.0,), (4.0,), (5.0,)]

pattern = midi.read_midifile(file('chpn_a_flat.mid'))
events = NoteEvents(pattern)
truth = events.get_ground_truth(31.25)
print truth.shape
io_utils.predictionToBasicMIDI(truth, 'testingmidi.mid')
