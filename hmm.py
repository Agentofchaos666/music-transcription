from collections import defaultdict

# faux pre-start bucket indices
START_0 = 0
START_1 = 1

class HMM():
    def __init__(self, numBuckets, nGramLength=2, laplace=0):
        self.numBuckets = numBuckets
        self.n = nGramLength
        # self.startCounts = [laplace for _ in range(numBuckets)] 
        # self.transCounts = [[laplace for _ in range(numBuckets)] for _ in range(numBuckets)]
        # self.emissionCounts = [[laplace for _ in range(numBuckets)] for _ in range(numBuckets)]
        self.transCounts = defaultdict(lambda: [laplace] * numBuckets)
        self.emissionCounts = defaultdict(lambda: [laplace] * numBuckets)

    def train(self, H, M):
        numExamples = len(H)
        for i in range(numExamples):
            h = H[i]
            e = E[i]
            numEvents = len(h) 
            for j in range(numEvents):
                if j == 0:
                    to_condition_on = (START_0, START_1)
                elif j == 1:
                    to_condition_on = (START_1, h[0])
                else:
                    to_condition_on = (h[j-2], h[j-1])
                self.transCounts[to_condition_on][h[j]] += 1
                self.emissionCounts[h[j]][e[j]] += 1

        trans_sums = {key : sum(counts) for key, counts in self.transCounts.iteritems()}
        self.transProbs = {key : [float(count)/trans_sums[key] for count in counts] for key, counts in self.transCounts.iteritems()}

        emission_sums = {key : sum(counts) for key, counts in self.emissionCounts.iteritems()}
        self.emissionProbs = {key : [float(count)/emission_sums[key] for count in counts] for key, counts in self.emissionCounts.iteritems()}

# model = HMM(11, 2, 1)
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

