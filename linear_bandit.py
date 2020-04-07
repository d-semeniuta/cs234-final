import torch
import torch.utils.data as data

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pdb

from util.data import WarfarinData, getNumFeatures
import util.utils as utils
from baselines import runBaselines

class LinUCB:
    def __init__(self, num_features, num_arms=3, delta=0.1):
        self.d = num_features
        self.K = num_arms
        self.As = []
        self.bs = []
        self.alpha = 1 + np.sqrt(np.log(2 / delta) / 2)
        for a in range(self.K):
            A = torch.eye(self.d)
            b = torch.zeros([self.d, 1])
            self.As.append(A)
            self.bs.append(b)

    def take_action(self, features):
        ps = []
        for A, b in zip(self.As, self.bs):
            A_inverse = torch.inverse(A)
            theta = torch.matmul(A_inverse, b)
            p_lhs = torch.matmul(theta.t(), features)
            p_rhs = torch.sqrt(torch.matmul(torch.matmul(features.t(), A_inverse), features))
            p = p_lhs + self.alpha * p_rhs
            ps.append(p)

        a_t = np.argmax(ps)
        return a_t

    def observe_reward(self, features, action_taken, reward):
        a_t = action_taken
        self.As[a_t] = torch.addr(self.As[a_t], features.squeeze(), features.squeeze())
        self.bs[a_t] += reward * features

def trainModel(model, bestLinear, warfarin_data, shuffle=False):
    # warfarin_data = WarfarinData('./data/warfarin.csv', item_type=feat_type)
    loader = data.DataLoader(warfarin_data, batch_size=1, shuffle=shuffle)
    accCounter = utils.AccuracyCounter()
    cum_regret, cum_reward = 0, 0
    num_regrets = 0
    running_err, running_regret, running_reward = [], [], []
    with torch.no_grad(), tqdm(total=len(loader.dataset)) as progress_bar:
        for i, (features_dict, features_array, target_dosage) in enumerate(loader):
            features = features_array
            features = features.type(torch.FloatTensor).t()
            pred = model.take_action(features)
            if pred == target_dosage:
                reward = 0
                accCounter.addCounts(num_correct=1)
            else:
                reward = -1
                accCounter.addCounts(num_wrong=1)
            model.observe_reward(features, pred, reward)
            regret = utils.calculateRegret(model, bestLinear, features.t(), pred).item()
            cum_regret += regret
            cum_reward += reward
            err = accCounter.getError()
            running_err.append(err)
            running_regret.append(cum_regret)
            running_reward.append(cum_reward)
            progress_bar.update(1)
            progress_bar.set_postfix(err=err, cum_reg=cum_regret, cum_rew=cum_reward)
    return running_err, running_regret, running_reward

def runLinBanditTrials(bestLinear, data, num_features, num_trials=20):
    print('Running {} Trials'.format(num_trials))
    results = {'err' : [], 'regret' : [], 'reward' : []}
    for _ in range(num_trials):
        model = LinUCB(num_features)
        running_err, running_regret, running_reward = trainModel(model, bestLinear, data, shuffle=True)
        results['err'].append(running_err)
        results['regret'].append(running_regret)
        results['reward'].append(running_reward)

    return results

def plotAccRegret(running_accs, running_regrets, baselines):
    """ DEPRECATED

    """
    running_accs = np.stack(running_accs)
    running_errs = 1 - running_accs
    running_regrets = np.stack(running_regrets)
    err_means, err_intervals = utils.getTIntervals(running_errs)
    reg_means, reg_intervals = utils.getTIntervals(running_regrets)
    num_samples = len(err_means)
    x = np.arange(1, num_samples+1)
    print('Plotting Error')
    plt.plot(x, err_means, label='Linear UCB')
    # plt.plot(x, acc_means, label='Linear UCB')
    plt.fill_between(x, err_intervals[0], err_intervals[1])

    plt.plot(x, [1 - baselines['linear'][0][-1]] * num_samples, label='Linear Fit')
    plt.plot(x, [1 - baselines['fixed'][0][-1]] * num_samples, label='Fixed Dose')
    plt.plot(x, [1 - baselines['clinical'][0][-1]]  * num_samples, label='Clinical Dose')
    plt.xlabel('Patient Samples Seen')
    plt.ylabel('Error Rate')
    plt.title('LinUCB Running Error vs Baseline Performance')
    plt.legend()
    plt.savefig('./plots/err.png')
    plt.clf()

    print('Plotting Regret')
    plt.plot(x, reg_means, label='Linear UCB')
    plt.fill_between(x, reg_intervals[0], reg_intervals[1])

    plt.plot(x, baselines['linear'][1], label='Linear Fit')
    plt.plot(x, baselines['fixed'][1], label='Fixed Dose')
    plt.plot(x, baselines['clinical'][1], label='Clinical Dose')
    plt.xlabel('Patient Samples Seen')
    plt.ylabel('Cumulative Regret')
    plt.title('Model Cumulative Regret Comparison')
    plt.legend()
    plt.savefig('./plots/regret.png')
    plt.clf()

def main():
    bestLinear, baseline_results = runBaselines()

    print('Training linear')
    warfarin_data = WarfarinData('./data/warfarin.csv', item_type='array')
    _, features, _ = warfarin_data[0]
    num_features = features.shape[0]
    running_accs, running_regrets = runLinBanditTrials(bestLinear, warfarin_data, num_features)
    plotAccRegret(running_accs, running_regrets, baseline_results)


if __name__ == '__main__':
    main()
