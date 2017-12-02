from collections import defaultdict
import itertools

class HMM():
    def __init__(self, buckets, nGramLength=2, laplace=1):
        self.numBuckets = len(buckets) 
        self.buckets = buckets
        self.n = nGramLength - 1 # number of things to condition on
        self.starts = [0] * self.n
        self.tCounts = defaultdict(lambda: defaultdict(lambda: laplace))
        self.eCounts = defaultdict(lambda: defaultdict(lambda: laplace))

    def train(self, H, E):
        numExamples = len(H)
        for i in range(numExamples):
            h = H[i]
            e = E[i]
            numEvents = len(h) 
            for j in range(numEvents):
                if j < self.n - 1:
                    to_condition_on = tuple(self.starts[j:j+self.n] + h[:j])
                else:
                    to_condition_on = tuple(h[j-self.n:j])
                # print h[j], to_condition_on
                self.tCounts[to_condition_on][h[j]] += 1
                self.eCounts[h[j]][e[j]] += 1

        possible_tuples = list(itertools.product(self.buckets, repeat=self.n))
        for i in range(len(possible_tuples)):
            possible_tuples[i] = tuple(possible_tuples[i])
        # print possible_tuples
        trans_sums = {key : sum([self.tCounts[key][bucket] for bucket in self.buckets]) for key in possible_tuples}
        self.transProbs = {key : {bucket : float(self.tCounts[key][bucket])/trans_sums[key] for bucket in self.buckets} for key in possible_tuples}

        emission_sums = {key : sum([self.eCounts[key][bucket] for bucket in self.buckets]) for key in self.buckets}
        self.emissionProbs = {key : {bucket : float(self.eCounts[key][bucket])/emission_sums[key] for bucket in self.buckets} for key in self.buckets}

    def predict(self, M):
        pass


# TESTING STUFF
# model = HMM(range(11), 4, 1)
# H = [[i] * 20 for i in range(10)]
# E = [[i+1] * 20 for i in range(10)]
# for i in range(10):
#     print 'H:', H[i]
#     print 'E:', E[i]
#     print '====================='
# model.train(H, E)
# for key, prob in model.transProbs.iteritems():
#     print key, ':', prob
# print '====================='
# for key, prob in model.emissionProbs.iteritems():
#     print key, ':', prob

