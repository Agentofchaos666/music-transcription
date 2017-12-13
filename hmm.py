from collections import defaultdict
import itertools

class HMM():
    def __init__(self, buckets, nGramLength=2, laplace=1):
        self.numBuckets = len(buckets) 
        self.buckets = buckets
        self.n = nGramLength - 1 # number of things to condition on
        self.starts = [None] * self.n
        self.tCounts = defaultdict(lambda: defaultdict(lambda: laplace))
        self.eCounts = defaultdict(lambda: defaultdict(lambda: laplace))

    def train(self, H, E):
        numExamples = len(H)
        for i in range(numExamples):
            h = H[i]
            e = E[i]
            numEvents = len(h) 
            for j in range(numEvents):
                if j < self.n:
                    to_condition_on = tuple(self.starts[j:j+self.n] + h[:j])
                else:
                    to_condition_on = tuple(h[j-self.n:j])
                # print h[j], to_condition_on
                self.tCounts[to_condition_on][h[j]] += 1
                self.eCounts[h[j]][e[j]] += 1

        possible_non_start_tuples = self._generate_non_start_conditions()
        possible_start_tuples = self._generate_start_conditions()
        possible_tuples = possible_non_start_tuples + possible_start_tuples
        print possible_start_tuples
        print '******************************************************'

        # print possible_tuples
        trans_sums = {key : sum([self.tCounts[key][bucket] for bucket in self.buckets]) for key in possible_tuples}
        self.transProbs = {key : {bucket : float(self.tCounts[key][bucket])/trans_sums[key] for bucket in self.buckets} for key in possible_tuples}

        emission_sums = {key : sum([self.eCounts[key][bucket] for bucket in self.buckets]) for key in self.buckets}
        self.emissionProbs = {key : {bucket : float(self.eCounts[key][bucket])/emission_sums[key] for bucket in self.buckets} for key in self.buckets}

    def _generate_non_start_conditions(self):
        # returns list of [(x_i-1, x_i-2, ... , x_n)] possible tuples to be conditioned
        # on not including special start state
        possible_tuples = list(itertools.product(self.buckets, repeat=self.n))
        for i in range(len(possible_tuples)):
            possible_tuples[i] = tuple(possible_tuples[i])
        return possible_tuples

    def _generate_start_conditions(self):
        # returns list of [(x_i-1, x_i-2, ... , x_n)] possible tuples that contain
        # special start state to be conditioned
        start_possible_tuples = []
        for j in range(self.n):
            if j < 1:
                start_possible_tuples.append(tuple(self.starts[j:j+self.n]))
            else:
                for permutation in itertools.product(self.buckets, repeat=j):
                    start_possible_tuples.append(tuple(self.starts[j:j+self.n] + list(permutation)))
        return start_possible_tuples

    def predict(self, E):
        predictions = []
        for i, e in enumerate(E):
            predictions.append(self.vitterbi(e))
            if i % 5 == 0:
                print str(i) + '/' + str(len(E))
        return predictions

    def vitterbi(self, E):
        '''
        Returns most likely sequence X of hidden states given model and E
        Handles nth order HMM's
        '''
        # states (x_t-n, x_t-n+1, ... , x_t-1, x_t)
        state_space = self._generate_non_start_conditions()
        obs_space = self.buckets
        K = len(self.buckets)   # size of observation space
        S = len(state_space)    # size of state space
        T = len(E)              # number of observations
        # dp is a S x T table w/ 
        # dp[i][j] = (prob of max prob path x_0, ... , x_j=state_space[i], 
        #             x_j-1 for max prob path)
        dp = [[(0,0)] * T for _ in range(S)]
        print 'dp dims:', len(dp), 'x', len(dp[0])
        print 'S =', S
        for i in range(S):
            bucket = state_space[i][-1]
            dp[i][0] = (self.transProbs[tuple(self.starts)][bucket] \
                        * self.emissionProbs[bucket][E[0]] ** 2, 0)

        # consistent indices contains indeces of conditions such that
        # k | state_space[i] can have a nonzero transition probability
        consistent_indices = {}
        for i in range(len(state_space)):
            indices = []
            key = state_space[i]
            # print key
            for j in range(len(state_space)):
                condition = state_space[j]
                # print 'condition', condition
                # print key[:self.n-1], condition[-(self.n-1):]
                if key[:self.n-1] == condition[-(self.n-1):]:
                    # print key[:self.n-1], condition[-(self.n-1):]
                    indices.append(j)
            consistent_indices[key] = indices

        for j in range(1, T):
            for i in range(S):
                curr_state = state_space[i]
                bucket = curr_state[-1]
                max_prob = -1
                max_prev_index = 0
                for k in consistent_indices[curr_state]:
                    prob = self.transProbs[tuple(state_space[k])][bucket] \
                            * dp[k][j-1][0] * self.emissionProbs[bucket][E[j]] ** 2
                    if prob > max_prob:
                        max_prob = prob
                        max_prev_index = k
                dp[i][j] = (max_prob, max_prev_index)

        # X = list of optimal hidden events
        X = [0] * T
        max_final_index = max(range(S), key=lambda x: dp[x][-1][0])
        X[-1] = state_space[max_final_index][-1]
        prev_index = dp[max_final_index][-1][1]
        for i in reversed(range(T-1)):
            X[i] = state_space[prev_index][-1]
            prev_index = dp[prev_index][-1][1]
        return X





    def predict_bigram(self, E):
        predictions = []

        for j,e in enumerate(E): 
            print j
            F = []
            numEvents = len(e)
            for i in range(numEvents):
                f = {}
                for bucket in self.buckets:
                    if i < self.n:
                        f[bucket] = self.transProbs[(self.starts[0],)][bucket] * self.emissionProbs[bucket][e[i]]
                    else: 
                        f[bucket] = sum([F[i-1][prev] * self.transProbs[(prev,)][bucket] * self.emissionProbs[bucket][e[i]] for prev in self.buckets])
                F.append(f)
            
            B = []
            for i in reversed(range(numEvents)):
                b = {}
                for bucket in self.buckets:
                    if i == numEvents - 1:
                        b[bucket] = 1. / self.numBuckets * self.emissionProbs[bucket][e[i]]
                    else:
                        b[bucket] = sum([B[numEvents-2-i][next_bucket] * self.transProbs[(bucket,)][next_bucket] * self.emissionProbs[bucket][e[i]] for next_bucket in self.buckets])
                B.append(b)

            S = [{bucket : F[i][bucket] * B[i][bucket] for bucket in self.buckets} for i in range(numEvents)]

            prediction = [max(S[i], key=S[i].get) for i in range(numEvents)]
            predictions.append(prediction)

        print E[0][:10]
        print '======================================'
        print predictions[0][:10]

        return predictions

# TESTING STUFF
# model = HMM(range(11), 4, 1)
# H = [[i] * 20 for i in range(10)]
# E = [[i+1] * 20 for i in range(10)]
# for i in range(10):
#     print 'H:', H[i]
#     print 'E:', E[i]
#     print '====================='
# model.train(H, E)
# predictions = model.predict(E)
# print predictions
# for key, prob in model.transProbs.iteritems():
#     print key, ':', prob
# print '====================='
# for key, prob in model.emissionProbs.iteritems():
#     print key, ':', prob

