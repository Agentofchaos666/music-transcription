# H = [true (duration, isNote) values]
# E = [observed (duration, isNote) values]

class HMM():
    def __init__(self, numBuckets, laplace=0):
        self.startCounts = [laplace for _ in range(numBuckets)] 
        self.transCounts = [[laplace for _ in range(numBuckets)] for _ in range(numBuckets)]
        self.emissionCounts = [[laplace for _ in range(numBuckets)] for _ in range(numBuckets)]

    def train(self, H, M):
        numExamples = len(H)
        for i in range(numExamples):
            h = H[i]
            e = E[i]
            numEvents = len(h) 
            self.startCounts[h[0]] += 1
            self.emissionCounts[h[0]][e[0]] += 1
            for j in range(numEvents - 1):
                self.transCounts[h[j-1]][h[j]] += 1
                self.emissionCounts[h[j]][e[0]] += 1

        start_sum = sum(self.startCounts)
        self.startProbs = [float(count)/sum for count in self.startCounts]

        trans_sums = [sum(counts) for counts in self.transCounts]
        self.transCounts = [[float(count)/trans_sums[i] for count in counts] for i, counts in enumerate(self.transCounts)]

        emission_sums = [sum(counts) for counts in self.emissionCounts]
        self.emissionCounts = [[float(count)/emission_sums[i] for count in counts] for i, counts in enumerate(self.emissionCounts)]

