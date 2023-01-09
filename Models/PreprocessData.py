from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
#
# Load AutoReg class from statsmodels.tsa.ar_model module
#
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    # select features we want
    '''
    features = ['week_start_date', 'reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']
    df = df[features]
    '''

    # fill missing values
    df.interpolate(option = 'spline', inplace = True)

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
testExog = pd.read_csv('../dengue_features_test.csv', index_col=[0, 1, 2])
sjTest = testExog.loc['sj']
iqTest = testExog.loc['iq']


sjTest.set_index(pd.to_datetime(sjTest['week_start_date'], format='%Y-%m-%d'), inplace=True)
sjTest.index = pd.DatetimeIndex(sjTest.index)
sjTest.index = pd.DatetimeIndex(sjTest.index).to_period('W')
sjTest.sort_index(inplace=True)

iqTest.set_index(pd.to_datetime(iqTest['week_start_date'], format='%Y-%m-%d'), inplace=True)
iqTest.index = pd.DatetimeIndex(iqTest.index)
iqTest.index = pd.DatetimeIndex(iqTest.index).to_period('W')
iqTest.sort_index(inplace=True)

#nothing missing in test data


df = pd.DataFrame(np.nan, index=[datetime.strptime('1995-12-25', '%Y-%m-%d'), datetime.strptime('2000-12-25', '%Y-%m-%d'), datetime.strptime('2006-12-25', '%Y-%m-%d')], columns=sj_train.columns)
sj_train.set_index(pd.to_datetime(sj_train['week_start_date'], format='%Y-%m-%d'), inplace = True)
sj_train.index = pd.DatetimeIndex(sj_train.index)
sj_train = pd.concat([sj_train, df])
sj_train.index = pd.DatetimeIndex(sj_train.index).to_period('W') #period is one week
sj_train.sort_index(inplace = True)
sj_train.interpolate(option = 'spline', inplace=True)
sj_train.fillna(method='ffill', inplace=True)


df = pd.DataFrame(np.nan, index=[datetime.strptime('2000-12-31', '%Y-%m-%d'), datetime.strptime('2006-12-31', '%Y-%m-%d')], columns=iq_train.columns)
iq_train.set_index(pd.to_datetime(iq_train['week_start_date'], format='%Y-%m-%d'), inplace = True)
iq_train.index = pd.DatetimeIndex(iq_train.index)
iq_train = pd.concat([iq_train, df])
iq_train.index = pd.DatetimeIndex(iq_train.index).to_period('W') #period is one week
iq_train.sort_index(inplace = True)
iq_train.interpolate(option = 'spline', inplace=True)
iq_train.fillna(method='ffill', inplace=True)



sj_train['4yearsAgo'] = sj_train['total_cases'].shift(260, axis = 0)

sj_train['month'] = int(sj_train.index.month[0])
sj_train['month_sin'] = np.sin(sj_train['month']/(12 * 2 * np.pi))
sj_train['month_cos'] = np.cos(sj_train['month']/(12 * 2 * np.pi))


sj_train['week'] = int(sj_train.index.week[0])
sj_train["week_sin"] = np.sin(sj_train['week']/(52 * 2 * np.pi))
sj_train["week_cos"] = np.cos(sj_train['week']/(52 * 2 * np.pi))

sj_train.drop(['month','week'],axis=1, inplace = True)


iq_train['4yearsAgo'] = iq_train['total_cases'].shift(208, axis = 0)

iq_train['month'] = int(iq_train.index.month[0])
iq_train['month_sin'] = np.sin(iq_train['month']/(12 * 2 * np.pi))
iq_train['month_cos'] = np.cos(iq_train['month']/(12 * 2 * np.pi))


iq_train['week'] = int(iq_train.index.week[0])
iq_train["week_sin"] = np.sin(iq_train['week']/(52 * 2 * np.pi))
iq_train["week_cos"] = np.cos(iq_train['week']/(52 * 2 * np.pi))

iq_train.drop(['month','week'],axis=1, inplace = True)

submission = pd.read_csv("../data-processed/submission_format (1).csv",
                         index_col=[0, 1, 2])

sj_train_exog = sj_train.drop(['total_cases', 'week_start_date',], axis = 1)
iq_train_exog = iq_train.drop(['total_cases', 'week_start_date'])