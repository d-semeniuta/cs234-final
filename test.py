import torch
import torch.utils.data as data

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import pdb

from util.data import WarfarinData
import util.utils as utils

from linear_bandit import runLinBanditTrials
from supervised_bandit import runSupBanditTrials
from baselines import runBaselines

def plotResults(baseline_results, model_results):
    """ Plot Error and Regret Results

    Parameters
    ----------
    baseline_results : dict
        dictionary of baseline results
            keys: 'linear', 'fixed', 'clinical'
            values: (running_acc, running_err)
    model_results : dict
        dictionary of model results
            keys: model run names
            values: (running_accs, running_errs) over 20 trials

    Returns
    -------
    type
        Description of returned object.

    """
    model_intervals = {}
    for model, results in model_results.items():
        running_errs = np.stack(results['err'])
        running_regrets = np.stack(results['regret'])
        running_rewards = np.stack(results['reward'])
        model_intervals[model] = {}
        model_intervals[model]['error'] = utils.getTIntervals(running_errs)
        model_intervals[model]['regret'] = utils.getTIntervals(running_regrets)
        model_intervals[model]['reward'] = utils.getTIntervals(running_rewards)

    num_samples = len(baseline_results['linear']['err'])
    x = np.arange(1, num_samples+1)

    print('Plotting Error')
    for model, pairs in model_intervals.items():
        means, intervals = pairs['error']
        plt.plot(x, means, label=model)
        plt.fill_between(x, intervals[0], intervals[1])
        print('Final Error for {}: {}'.format(model, means[-1]))

    plt.plot(x, [baseline_results['linear']['err'][-1]] * num_samples, label='Linear Fit')
    plt.plot(x, [baseline_results['fixed']['err'][-1]] * num_samples, label='Fixed Dose')
    plt.plot(x, [baseline_results['clinical']['err'][-1]]  * num_samples, label='Clinical Dose')
    plt.xlabel('Patient Samples Seen')
    plt.ylabel('Error Rate')
    plt.title('Model Running Errors vs Baseline Performance')
    plt.legend()
    plt.savefig('./plots/combined_err.png')
    plt.clf()

    print('Plotting Regret')
    for model, pairs in model_intervals.items():
        means, intervals = pairs['regret']
        plt.plot(x, means, label=model)
        plt.fill_between(x, intervals[0], intervals[1])
        print('Final Regret for {}: {}'.format(model, means[-1]))

    plt.plot(x, baseline_results['linear']['regret'], label='Linear Fit')
    plt.plot(x, baseline_results['fixed']['regret'], label='Fixed Dose')
    plt.plot(x, baseline_results['clinical']['regret'], label='Clinical Dose')
    plt.xlabel('Patient Samples Seen')
    plt.ylabel('Cumulative Regret')
    plt.title('Model Cumulative Regret Comparison')
    plt.legend()
    plt.savefig('./plots/combined_regret.png')
    plt.clf()

    print('Plotting Reward')
    for model, pairs in model_intervals.items():
        means, intervals = pairs['reward']
        plt.plot(x, means, label=model)
        plt.fill_between(x, intervals[0], intervals[1])
        print('Final Reward for {}: {}'.format(model, means[-1]))

    plt.plot(x, baseline_results['linear']['reward'], label='Linear Fit')
    plt.plot(x, baseline_results['fixed']['reward'], label='Fixed Dose')
    plt.plot(x, baseline_results['clinical']['reward'], label='Clinical Dose')
    plt.xlabel('Patient Samples Seen')
    plt.ylabel('Cumulative Reward')
    plt.title('Model Cumulative Reward Comparison')
    plt.legend()
    plt.savefig('./plots/combined_reward.png')
    plt.clf()

def main():
    print('Running Baselines')
    bestLinear, baseline_results = runBaselines()

    model_results = {}
    warfarin_data = WarfarinData('./data/warfarin.csv', item_type='array')
    _, features, _ = warfarin_data[0]
    num_features = features.shape[0]
    print('\nTraining LinUCB Bandit')
    model_results['LinUCB'] = runLinBanditTrials(bestLinear, warfarin_data, num_features)

    print('\nTraining Supervised Bandit')
    model_results['SupBandit'] = runSupBanditTrials(bestLinear, warfarin_data)

    print('\nPlotting Results')
    plotResults(baseline_results, model_results)



if __name__ == '__main__':
    main()
