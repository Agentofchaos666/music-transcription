# H = [true (duration, isNote) values]
# E = [observed (duration, isNote) values]

class HMM():
    def __init__(self, numBuckets, laplace=0):
        self.start_counts = [[laplace for _ in range(numBuckets)] for _ in range(numBuckets)]
        self.start_counts = 

    def train(self, H, M):
        numExamples = len(H)
        for i in range(numExamples):
