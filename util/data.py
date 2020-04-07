import numpy as np
import pandas as pd

import torch
import torch.utils.data as data

import pdb


cols_to_use = [
    'gender_m',
    'gender_f',
    'gender_u',
    'race_white',
    'race_unknown',
    'race_asian',
    'race_black',
    'ethnicity_hispanic',
    'ethnicity_nothispanic',
    'ethnicity_unknown',
    'age_in_decades',
    'height_cm',
    'weight_kg',
    'enzyme_inducer',
    'amiadarone',
    'vkorc1_ga',
    'vkorc1_aa',
    'vkorc1_unknown',
    'cyp2c9_12',
    'cyp2c9_13',
    'cyp2c9_22',
    'cyp2c9_23',
    'cyp2c9_33',
    # 'Therapeutic Dose of Warfarin',
    'dosage_level'
]

class WarfarinData(data.Dataset):
    """ Warfarin Dataset generated from warfarin.csv

    See project appendix for dataset info


    """
    def __init__(self, csv_loc, item_type='array'):
        super(WarfarinData, self).__init__()

        if item_type not in ['dict', 'array']:
            raise ValueError('Item type not supported: {}'.format(item_type))

        df = pd.read_csv(csv_loc)
        self.df = cleanWarfarinData(df)
        self.item_type = item_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # if self.item_type == 'dict':
        #     x = self.df.loc[:, self.df.columns != 'dosage_level'].iloc[idx].to_dict()
        # elif self.item_type == 'array':
        #     x = self.df.loc[:, self.df.columns != 'dosage_level'].iloc[idx].to_numpy()
            # x = torch.tensor(x)
        x = self.df.loc[:, self.df.columns != 'dosage_level'].iloc[idx]
        y = self.df.iloc[idx]['dosage_level']
        return x.to_dict(), x.to_numpy(), y

def cleanWarfarinData(df):
    df.dropna(subset=['Therapeutic Dose of Warfarin'], inplace=True)
    df2 = df.copy(deep=True)

    # cols_to_use = [
    #     'gender_m',
    #     'gender_f',
    #     'gender_u',
    #     'race_white',
    #     'race_unknown',
    #     'race_asian',
    #     'race_black',
    #     'ethnicity_hispanic',
    #     'ethnicity_nothispanic',
    #     'ethnicity_unknown',
    #     'age_in_decades',
    #     'height_cm',
    #     'weight_kg',
    #     'enzyme_inducer',
    #     'amiadarone',
    #     'vkorc1_ga',
    #     'vkorc1_aa',
    #     'vkorc1_unknown',
    #     'cyp2c9_12',
    #     'cyp2c9_13',
    #     'cyp2c9_22',
    #     'cyp2c9_23',
    #     'cyp2c9_33',
    #     # 'Therapeutic Dose of Warfarin',
    #     'dosage_level'
    # ]
    df2['gender_m'] = np.where(df['Gender']=='male', 1, 0)
    df2['gender_f'] = np.where(df['Gender']=='female', 1, 0)
    df2['gender_u'] = np.where(df['Gender']!=df['Gender'], 1, 0)

    df2['race_white'] = np.where(df['Race']=='White', 1, 0)
    df2['race_unknown'] = np.where((df['Race']=='Unknown') | (df['Race'].isna()), 1, 0)
    df2['race_asian'] = np.where(df['Race']=='Asian', 1, 0)
    df2['race_black'] = np.where(df['Race']=='Black or African American', 1, 0)

    df2['ethnicity_hispanic'] = np.where(df['Ethnicity']=='Hispanic or Latino', 1, 0)
    df2['ethnicity_nothispanic'] = np.where(df['Ethnicity']=='not Hispanic or Latino', 1, 0)
    df2['ethnicity_unknown'] = np.where(df['Ethnicity']!=df['Ethnicity'], 1, 0)

    df['Age'].fillna('0', inplace=True)
    age_in_decades = df['Age'].str[0].astype(int)
    age_in_decades = age_in_decades.replace(0, np.NaN)
    mean_age = age_in_decades.mean()
    df2['age_in_decades'] = age_in_decades.replace(np.NaN, mean_age)
    # df['age_in_decades'] = np.where(df['Age'].isna(), 0, df['Age'].str[0].astype(int))

    df2['height_cm'] = df['Height (cm)']
    mean_height = df2['height_cm'].mean()
    df2['height_cm'].fillna(mean_height, inplace=True)
    df2['weight_kg'] = df['Weight (kg)']
    mean_weight = df2['weight_kg'].mean()
    df2['weight_kg'].fillna(mean_weight, inplace=True)

    df2['enzyme_inducer'] = np.where(
        (df['Carbamazepine (Tegretol)'] == 1) | (df['Phenytoin (Dilantin)'] == 1) | (df['Rifampin or Rifampicin'] == 1), 1, 0
    )

    df2['amiadarone'] = np.where(df['Amiodarone (Cordarone)'] == 1, 1, 0)

    df2['vkorc1_ga'] = np.where(df['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T']=='A/G', 1, 0)
    df2['vkorc1_aa'] = np.where(df['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T']=='A/A', 1, 0)
    df2['vkorc1_unknown'] = np.where(df['VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T'].isna(), 1, 0)

    df2['cyp2c9_12'] = np.where(df['Cyp2C9 genotypes']=='*1/*2', 1, 0)
    df2['cyp2c9_13'] = np.where(df['Cyp2C9 genotypes']=='*1/*3', 1, 0)
    df2['cyp2c9_22'] = np.where(df['Cyp2C9 genotypes']=='*2/*2', 1, 0)
    df2['cyp2c9_23'] = np.where(df['Cyp2C9 genotypes']=='*2/*3', 1, 0)
    df2['cyp2c9_33'] = np.where(df['Cyp2C9 genotypes']=='*3/*3', 1, 0)

    dosage_conditions = [
        df['Therapeutic Dose of Warfarin'] < 21,
        (df['Therapeutic Dose of Warfarin'] >= 21) & (df['Therapeutic Dose of Warfarin'] <= 49),
        df['Therapeutic Dose of Warfarin'] > 49,
    ]

    df2['dosage_level'] = np.select(dosage_conditions, [0, 1, 2])

    cleaned_df = df2[cols_to_use]
    return cleaned_df

def getNumFeatures():
    return len(cols_to_use) - 1
