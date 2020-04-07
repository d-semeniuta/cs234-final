import torch
import torch.utils.data as data

from tqdm import tqdm
import pdb

from util.data import WarfarinData
import util.utils as utils

def trainModel(model, feat_type='array', predict_raw_dose=False):
    warfarin_data = WarfarinData('./data/warfarin.csv', item_type=feat_type)
    loader = data.DataLoader(warfarin_data, batch_size=1)
    accCounter = utils.AccuracyCounter()
    with torch.no_grad(), tqdm(total=len(loader.dataset)) as progress_bar:
        for i, (features_dict, features_array, target_dosage) in enumerate(loader):
            pred = model.take_action(features)
            if pred == target_dosage:
                reward = 0
                accCounter.addCounts(num_correct=1)
            else:
                reward = -1
                accCounter.addCounts(num_wrong=1)
            model.observe_reward(features, pred, reward)
            progress_bar.update(batch_size)
            progress_bar.set_postfix(acc=accCounter.getAccuracy())

    return accCounter.getAccuracy()
