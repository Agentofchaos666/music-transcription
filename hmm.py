from collections import defaultdict

class HMM():
    def __init__(self, numBuckets, nGramLength=2, laplace=0):
        self.numBuckets = numBuckets
        self.n = nGramLength - 1
        self.starts = [0] * self.n
        self.transCounts = defaultdict(lambda: [laplace] * numBuckets)
        self.emissionCounts = defaultdict(lambda: [laplace] * numBuckets)

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
                self.transCounts[to_condition_on][h[j]] += 1
                self.emissionCounts[h[j]][e[j]] += 1

        trans_sums = {key : sum(counts) for key, counts in self.transCounts.iteritems()}
        self.transProbs = {key : [float(count)/trans_sums[key] for count in counts] for key, counts in self.transCounts.iteritems()}

        emission_sums = {key : sum(counts) for key, counts in self.emissionCounts.iteritems()}
        self.emissionProbs = {key : [float(count)/emission_sums[key] for count in counts] for key, counts in self.emissionCounts.iteritems()}

    def predict(self, M):
        pass


# TESTING STUFF
# model = HMM(11, 4, 1)
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

