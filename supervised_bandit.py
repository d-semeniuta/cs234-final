
import torch
import torch.utils.data as data

from tqdm import tqdm
import numpy as np
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
import pdb

from util.data import WarfarinData, getNumFeatures
import util.utils as utils
from baselines import runBaselines


class SupervisedBandit:
    def __init__(self, num_arms=3):
        self.K = num_arms
        self.training_data = None
        self.training_labels = None
        self.clf = RidgeClassifier()
        self.dont_fit = True

    def take_action(self, features):
        if self.training_data is None:
            return torch.tensor(np.random.choice(self.K))
        elif not self.dont_fit: # don't fit until have enough unique classes
            return torch.tensor(self.clf.predict(features))
        else:
            return torch.tensor(self.training_labels[0])

    def add_data(self, features, correct_action):
        if self.training_data is None:
            self.training_data = features
            self.training_labels = np.array([correct_action])
        else:
            self.training_data = torch.cat((self.training_data, features))
            self.training_labels = np.concatenate((self.training_labels, correct_action))

        if len(np.unique(self.training_labels)) > 1:
            # solver needs at least 2 unique classes to fit
            self.dont_fit = False
            self.clf.fit(self.training_data, self.training_labels)

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
            pred = model.take_action(features)
            if pred == target_dosage:
                reward = 0
                accCounter.addCounts(num_correct=1)
            else:
                reward = -1
                accCounter.addCounts(num_wrong=1)
            model.add_data(features, target_dosage)
            regret = utils.calculateRegret(model, bestLinear, features, pred).item()

            cum_regret += regret
            cum_reward += reward
            err = accCounter.getError()
            running_err.append(err)
            running_regret.append(cum_regret)
            running_reward.append(cum_reward)
            progress_bar.update(1)
            progress_bar.set_postfix(err=err, cum_reg=cum_regret, cum_rew=cum_reward)

    return running_err, running_regret, running_reward

def runSupBanditTrials(bestLinear, data, num_trials=20):
    print('Running {} Trials'.format(num_trials))
    results = {'err' : [], 'regret' : [], 'reward' : []}
    for _ in range(num_trials):
        model = SupervisedBandit()
        running_err, running_regret, running_reward = trainModel(model, bestLinear, data, shuffle=True)
        results['err'].append(running_err)
        results['regret'].append(running_regret)
        results['reward'].append(running_reward)

    return results

def plotAccRegret(running_accs, running_regrets, baselines):
    running_accs = np.stack(running_accs)
    running_errs = 1 - running_accs
    running_regrets = np.stack(running_regrets)
    # acc_means, acc_intervals = utils.getTIntervals(running_accs)
    err_means, err_intervals = utils.getTIntervals(running_errs)
    reg_means, reg_intervals = utils.getTIntervals(running_regrets)
    num_samples = len(err_means)
    x = np.arange(1, num_samples+1)
    print('Plotting Error')
    plt.plot(x, err_means, label='SupervisedBandit')
    # plt.plot(x, acc_means, label='Linear UCB')
    plt.fill_between(x, err_intervals[0], err_intervals[1])

    plt.plot(x, [1 - baselines['linear'][0][-1]] * num_samples, label='Linear Fit')
    plt.plot(x, [1 - baselines['fixed'][0][-1]] * num_samples, label='Fixed Dose')
    plt.plot(x, [1 - baselines['clinical'][0][-1]]  * num_samples, label='Clinical Dose')
    plt.xlabel('Patient Samples Seen')
    plt.ylabel('Error Rate')
    plt.title('SupervisedBandit Running Error vs Baseline Performance')
    plt.legend()
    plt.savefig('./plots/err_sup.png')
    plt.clf()

    print('Plotting Regret')
    plt.plot(x, reg_means, label='SupervisedBandit')
    plt.fill_between(x, reg_intervals[0], reg_intervals[1])

    plt.plot(x, baselines['linear'][1], label='Linear Fit')
    plt.plot(x, baselines['fixed'][1], label='Fixed Dose')
    plt.plot(x, baselines['clinical'][1], label='Clinical Dose')
    plt.xlabel('Patient Samples Seen')
    plt.ylabel('Cumulative Regret')
    plt.title('Model Cumulative Regret Comparison')
    plt.legend()
    plt.savefig('./plots/regret_sup.png')
    plt.clf()

def main():
    bestLinear, baseline_results = runBaselines()

    print('Training linear')
    warfarin_data = WarfarinData('./data/warfarin.csv', item_type='array')
    running_accs, running_regrets = runSupBanditTrials(bestLinear, warfarin_data, num_trials=20)
    plotAccRegret(running_accs, running_regrets, baseline_results)

if __name__ == '__main__':
    main()
