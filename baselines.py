import torch
import torch.utils.data as data

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

from tqdm import tqdm
import pdb
import pickle

from util.data import WarfarinData, cleanWarfarinData
import util.utils as utils

class Linear:
    def __init__(self, csv_loc):
        df = pd.read_csv(csv_loc)
        cleaned_df = cleanWarfarinData(df)
        data = cleaned_df.loc[:, cleaned_df.columns != 'dosage_level'].to_numpy()
        labels = cleaned_df['dosage_level'].to_numpy()
        self.K = labels.max()+1
        rewards = np.zeros((labels.size, self.K)) - 1 # array of negative one
        rewards[np.arange(labels.size),labels] = 0
        # pdb.set_trace()
        self.models = []
        for a in range(self.K):
            this_model = LinearRegression()
            this_model.fit(data, rewards[:,a])
            self.models.append(this_model)


    def get_linear_reward(self, features, action):
        # pdb.set_trace()
        if torch.is_tensor(action):
            action = int(action.item())
        # features = features.squeeze()
        return torch.tensor(self.models[action].predict(features))

    def take_action(self, features):
        rs = []
        for a in range(self.K):
            rs.append(self.get_linear_reward(features, a))
        rs = torch.stack(rs)
        a_star = torch.argmax(rs, 0)
        return a_star

class FixedDose:
    def __init__(self):
        self.dose = 35

    def take_action(self, features):
        batch_size = len(features['age_in_decades'])
        dose = [self.dose] * batch_size
        return torch.tensor(dose)

class ClinicalDose:
    def __init__(self):
        self.dose = 35

    def take_action(self, features):
        """ Given dictionary of features,

        """
        batch_size = len(features['age_in_decades'])
        dose = torch.zeros([batch_size], dtype=torch.float)
        dose += 4.0376
        dose -= 0.2546 * features['age_in_decades']
        dose += 0.0118 * features['height_cm']
        dose += 0.0134 * features['weight_kg']
        dose -= 0.6752 * features['race_asian']
        dose += 0.5060 * features['race_black']
        dose += 0.0443 * features['race_unknown']
        dose += 1.2799 * features['enzyme_inducer']
        dose -= 0.5695 * features['amiadarone']
        return dose**2

def testBaseline(model, bestLinear, feat_type='array', predict_raw_dose=False):
    warfarin_data = WarfarinData('./data/warfarin.csv', item_type=feat_type)
    loader = data.DataLoader(warfarin_data, batch_size=1)
    accCounter = utils.AccuracyCounter()
    cum_regret, cum_reward = 0, 0
    running_err, running_regret, running_reward = [], [], []
    with torch.no_grad(), tqdm(total=len(loader.dataset)) as progress_bar:
        for i, (features_dict, features_array, target_dosage) in enumerate(loader):
            batch_size = target_dosage.shape[0]
            features = features_dict if feat_type=='dict' else features_array
            preds = model.take_action(features)
            preds_bins = utils.bin_dosage(preds) if predict_raw_dose else preds
            # pdb.set_trace()
            matches = len(torch.nonzero(target_dosage == preds_bins))
            reward = 0 if matches else -1
            accCounter.addCounts(num_correct=matches, num_wrong=(batch_size-matches))
            regret = utils.calculateRegret(model, bestLinear, features_array, preds_bins).item()

            cum_regret += regret
            cum_reward += reward
            err = accCounter.getError()
            running_err.append(err)
            running_regret.append(cum_regret)
            running_reward.append(cum_reward)
            progress_bar.update(batch_size)
            progress_bar.set_postfix(err=err, cum_reg=cum_regret, cum_rew=cum_reward)

    return accCounter.getAccuracy(), running_err, running_regret, running_reward

def runBaselines():
    results = {}
    print('Running Best Linear baseline')
    linear = Linear('./data/warfarin.csv')
    acc, running_err, running_regret, running_reward = testBaseline(linear, linear, feat_type='array')
    results['linear'] = {'err': running_err, 'regret' : running_regret, 'reward' : running_reward}
    print('Linear model Accuracy: {}\t Cumulative regret'.format(acc, running_regret[-1]))

    print('Running Fixed Dose baseline')
    fixedDose = FixedDose()
    acc, running_err, running_regret, running_reward = testBaseline(fixedDose, linear, feat_type='dict', predict_raw_dose=True)
    results['fixed'] = {'err': running_err, 'regret' : running_regret, 'reward' : running_reward}
    print('Fixed Dose Accuracy: {}\t Cumulative regret'.format(acc, running_regret[-1]))

    print('Running Clinical Dose baseline')
    clinicalDose = ClinicalDose()
    acc, running_err, running_regret, running_reward = testBaseline(clinicalDose, linear, feat_type='dict', predict_raw_dose=True)
    results['clinical'] = {'err': running_err, 'regret' : running_regret, 'reward' : running_reward}
    print('Clinical Dose: Accuracy: {}\t Cumulative regret'.format(acc, running_regret[-1]))

    return linear, results


def main():
    runBaselines()

if __name__ == '__main__':
    main()
