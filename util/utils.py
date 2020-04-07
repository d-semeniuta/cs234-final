import numpy as np
import torch
import scipy.stats

import pdb

def bin_dosage(dose):
    """ Convert raw dosage in mg/wk to bins
    """
    bins = torch.ones_like(dose)
    twos = torch.ones_like(dose) * 2
    bins = torch.where(dose < 21, torch.zeros_like(dose), bins)
    bins = torch.where(dose > 49, twos, bins)
    return bins
    # return np.select([dose < 21, dose > 49], [0, 2], default=1)

class AccuracyCounter():
    def __init__(self):
        self.num_correct, self.num_wrong = 0, 0

    def addCounts(self, num_correct=0, num_wrong=0):
        self.num_correct += num_correct
        self.num_wrong += num_wrong

    def getAccuracy(self):
        return self.num_correct / (self.num_correct + self.num_wrong)

    def getError(self):
        return self.num_wrong / (self.num_correct + self.num_wrong)


def getTIntervals(values, confidence=0.95):
    num_samples = values.shape[0]
    means = np.mean(values, axis=0)
    sstd = scipy.stats.sem(values, axis=0)
    intervals = scipy.stats.t.interval(confidence, num_samples, loc=means, scale=sstd)
    return means, intervals

def calculateRegret(model, bestLinear, features, action_taken):
    best_action = bestLinear.take_action(features)
    best_reward = bestLinear.get_linear_reward(features, best_action)
    observed_reward = bestLinear.get_linear_reward(features, action_taken)
    return best_reward - observed_reward

def testGetTIntervals():
    values = np.random.uniform(size=(20,30))
    getTIntervals(values)

def main():
    testGetTIntervals()

if __name__ == '__main__':
    main()
