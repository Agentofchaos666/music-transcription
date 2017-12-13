import midi
import math
import io_utils

TEST_DIRS = ['mozart']
TEST_BUCKETS = [(1.0/64,), (1.0/32,), (1.0/16, 'd'), (1.0/16, 't'), (1.0/16,),
                (1.0/8, 'd'), (1.0/8, 't'), (1.0/8,),
                (1.0/4, 'd'), (1.0/4, 't'), (1.0/4,),
                (1.0/2, 'd'), (1.0/2,),
                (1.0,), (2.0,), (3.0,), (4.0,), (5.0,)]
SIMPLE_BUCKETS = [(1.0/64,), (1.0/32,), (1.0/16, 'd'), (1.0/16, 't'), (1.0/16,),
                (1.0/8, 'd'), (1.0/8, 't'), (1.0/8,),
                (1.0/4, 'd'), (1.0/4, 't'), (1.0/4,),
                (1.0/2, 'd'), (1.0/2,),
                (1.0,)]
# TEST_BUCKETS = [(1.0/16,),
#                 (1.0/8, 'd'), (1.0/8, 't'), (1.0/8,),
#                 (1.0/4, 'd'), (1.0/4, 't'), (1.0/4,),
#                 (1.0/2, 'd'), (1.0/2,),
#                 (1.0,), (2.0,), (3.0,), (4.0,), (5.0,)]

OUTPUT_DIR = 'predicted_tempos/'

class TempoDetector:
    def __init__(self, buckets, ticks_per_beat=400):
        self.ticks_per_beat = ticks_per_beat
        self.setBuckets(buckets)
        print self.bucket_ticks

    def setBuckets(self, buckets):
        self.buckets = buckets
        self.bucket_ticks = [io_utils.bucketToTicks(bucket, self.ticks_per_beat) for bucket in self.buckets]

    def logDistanceFunction(self, event_tick, bucket_tick):
        return math.log(event_tick) - math.log(bucket_tick)

    def linearDistanceFunction(self, event_tick, bucket_tick):
        return event_tick - bucket_tick

    def linearDistanceFunctionGradient(self, event_tick, bucket_tick):
        return 1

    def predict(self, time_event_list, start_tempo, tick_distance_fn, tick_distance_grad_fn, stepfn):
        '''
        Input:
        time_event_list = [event_length, ...] in microseconds
        start_tempo in bpm
        tick_distance_fn(event_tick, bucket_tick) --> 'distance' between ticks
        tick_distance_grad_fn(event_tick, bucket_tick) is gradient of tick_distance_fn
        step_fn(iter) --> step size
        Output:
        predicted tempo from SGD
        '''
        tempo = start_tempo
        prev_loss = self.computeLoss(tempo, time_event_list, tick_distance_fn)
        # print 'Starting loss', prev_loss
        iteration = 0
        while True:
            iteration += 1
            for event_length in time_event_list:
                tempo -= stepfn(iteration) * \
                            self.computeStochasticGradient(tempo, event_length, tick_distance_fn, tick_distance_grad_fn)
            # print tempo
            curr_loss = self.computeLoss(tempo, time_event_list, tick_distance_fn)
            # print curr_loss
            if abs(prev_loss - curr_loss) < 1e-6:
                break
            prev_loss = curr_loss
        return (int(tempo), int(curr_loss))

    def mpqn_to_bpm(self, mpqn):
        return 6e7 / mpqn

    def bpm_to_mpqn(self, bpm):
        return 6e7 / bpm

    def timeToTick(self, time, tempo):
        # time in ms, tempo in bpm
        return self.ticks_per_beat * (1.0 / self.bpm_to_mpqn(tempo)) * time

    def timeToTickGradient(self, time, tempo):
        # gradient with respect to time
        return self.ticks_per_beat * (1.0 / self.bpm_to_mpqn(tempo))

    def computeLoss(self, tempo, time_event_list, tick_distance_fn):
        loss = 0
        for time_event in time_event_list:
            #print tick_distance_fn(100, 200)
            event_loss = min(tick_distance_fn(self.timeToTick(time_event, tempo), bucket_tick) ** 2\
                             for bucket_tick in self.bucket_ticks)
            loss += event_loss
        return loss

    def computeStochasticGradient(self, tempo, event_length, tick_distance_fn, tick_distance_grad_fn):
        loss_by_bucket = [tick_distance_fn(self.timeToTick(event_length, tempo), bucket_tick) ** 2\
                             for bucket_tick in self.bucket_ticks]
        optimal_bucket_index = min(range(len(loss_by_bucket)), key=lambda x: loss_by_bucket[x])
        event_tick = self.timeToTick(event_length, tempo)
        bucket_tick = self.bucket_ticks[optimal_bucket_index]
        return 2 * tick_distance_fn(event_tick, bucket_tick) * tick_distance_grad_fn(event_tick, bucket_tick) \
                    * self.timeToTickGradient(event_length, tempo)


if __name__ == '__main__':
    # for i, f in enumerate(filenames):
    #     print i, f
    # detector = TempoDetector(TEST_BUCKETS)
    # start_tempos = range(40, 200, 10)
    # candidates = [detector.predict(E_time_mat[3], t, detector.linearDistanceFunction, \
    #                     detector.linearDistanceFunctionGradient, lambda x: 1) for t in start_tempos]
    # print filenames[3]
    # print 'Candidates', candidates
    # print 'Median tempo', tempos[3]
    print TEST_BUCKETS
    E_time_mat, tempos, filenames = io_utils.generateTempoPredictionData(TEST_DIRS)
    detector = TempoDetector(TEST_BUCKETS)
    start_tempos = range(40, 200, 10)
    for i in range(1):
        candidates_losses = [detector.predict(E_time_mat[i], t, detector.linearDistanceFunction, \
                            detector.linearDistanceFunctionGradient, lambda x: 1) for t in start_tempos]
        candidates_losses = list(set(candidates_losses))

        candidates_linear = [c for c,l in candidates_losses]
        print filenames[i]
        print 'Candidates: ', sorted(candidates_losses, key=lambda x: x[1])
        print 'Median tempo: ', tempos[i]
        print 'Min offset linear: ', min(abs(c - tempos[i]) for c in candidates_linear)
        E_pred_mat, sigs = io_utils.generatePredictedTempoMatrix(filenames[i], TEST_BUCKETS, candidates_linear)
        # detector.setBuckets(SIMPLE_BUCKETS)
        for c in candidates_linear:
            print "Candidate", c
            print 'Loss: ', detector.computeLoss(c, E_time_mat[i], detector.logDistanceFunction)
        # print len(E_pred_mat)
        # print len(E_pred_mat[0])
        # print E_pred_mat[0][:5]
        # print E_pred_mat[-1][:5]
        # for i in range(len(candidates_linear)):
        #     io_utils.eventListToMIDI(E_pred_mat[i], TEST_BUCKETS, 400, \
        #         int(candidates_linear[i]), filenames[i][:-4] + '_' + str(candidates_linear[i]) + '.mid', OUTPUT_DIR, timeSig=sigs[i][0], keySig=sigs[i][1])
    # H_mat, E_mat, _, _, signatures = io_utils.generateTrainData(TEST_DIRS, TEST_BUCKETS, allowed_tempo_diff=float('inf'))
    # print filenames[0]
    # print len(E_time_mat)
    # print len(E_time_mat[0])
    # print len(E_mat)
    # print len(E_mat[0])
    # print E_time_mat[0][:5]
    # print E_mat[0][:5]
