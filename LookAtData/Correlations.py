import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
#
# Load AutoReg class from statsmodels.tsa.ar_model module
#
import seaborn as sns

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX


def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    # select features we want
    features = []
    df = df[features]

    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)

    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']

    return sj, iq

sj_train, iq_train = preprocess_data('../CSV/dengue_features_train.csv',
                                     labels_path="../CSV/dengue_labels_train.csv")


sj_train['1_years'] = sj_train['total_cases'].shift(52, axis = 0)
sj_train['2_years'] = sj_train['total_cases'].shift(104, axis = 0)
sj_train['3_years'] = sj_train['total_cases'].shift(156, axis = 0)
sj_train['4_years'] = sj_train['total_cases'].shift(208, axis = 0)
sj_train['5_years'] = sj_train['total_cases'].shift(260, axis = 0)
sj_train['6_years'] = sj_train['total_cases'].shift(312, axis = 0)
sj_train['7_years'] = sj_train['total_cases'].shift(364, axis = 0)
sj_train['8_years'] = sj_train['total_cases'].shift(416, axis = 0)
sj_train['9_years'] = sj_train['total_cases'].shift(468, axis = 0)
sj_train['10_years'] = sj_train['total_cases'].shift(520, axis = 0)
#sj_train.to_csv('tempdf1.csv')
'''
sj_train['1_days'] = sj_train['total_cases'].shift(1, axis = 0)
sj_train['2_days'] = sj_train['total_cases'].shift(2, axis = 0)
sj_train['3_days'] = sj_train['total_cases'].shift(3, axis = 0)
sj_train['4_days'] = sj_train['total_cases'].shift(4, axis = 0)
sj_train['5_days'] = sj_train['total_cases'].shift(5, axis = 0)
sj_train['6_days'] = sj_train['total_cases'].shift(6, axis = 0)
sj_train['7_days'] = sj_train['total_cases'].shift(7, axis = 0)
sj_train['8_days'] = sj_train['total_cases'].shift(8, axis = 0)
'''
sj_train.dropna(inplace = True)

sj_correlations = sj_train.corr()

sj_corr_heat = sns.heatmap(sj_correlations, annot = True, fmt = '.2f')

#sj_train['total_cases'].plot()

#plt.savefig('4YearCorrWithNum')
plt.show()