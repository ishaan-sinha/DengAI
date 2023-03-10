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

'''
from datetime import datetime, date, timedelta

df = pd.DataFrame(np.nan, index=[datetime.strptime('1995-12-25', '%Y-%m-%d'), datetime.strptime('2000-12-25', '%Y-%m-%d'), datetime.strptime('2006-12-25', '%Y-%m-%d')], columns=sj_train.columns)
sj_train.set_index(pd.to_datetime(sj_train['week_start_date'], format='%Y-%m-%d'), inplace = True)
sj_train.index = pd.DatetimeIndex(sj_train.index)
sj_train = pd.concat([sj_train, df])


sj_train.index = pd.DatetimeIndex(sj_train.index).to_period('W') #period is one week
sj_train.sort_index(inplace = True)
sj_train.interpolate(option = 'spline', inplace=True)
sj_train.fillna(method='ffill', inplace=True)

#Extremely janky but works
sj_train['last_year_cases'] = sj_train['total_cases'].shift(52, axis = 0)
sj_train['1w'] = sj_train['total_cases'].shift(1, axis = 0)
sj_train['2w'] = sj_train['total_cases'].shift(2, axis = 0)
sj_train['3w'] = sj_train['total_cases'].shift(3, axis = 0)
sj_train['last3WeekAverage'] = (sj_train['1w']+sj_train['2w']+sj_train['3w'])/3
sj_train.drop(['1w','2w','3w'], axis = 1)
sj_train['diff1-2'] = sj_train['1w'] - sj_train['2w']
sj_train['diff2-3'] = sj_train['2w'] - sj_train['3w']

sj_train['month'] = int(sj_train.index.month[0])
sj_train['month_sin'] = np.sin(sj_train['month']/(12 * 2 * np.pi))
sj_train['month_cos'] = np.cos(sj_train['month']/(12 * 2 * np.pi))


sj_train['week'] = int(sj_train.index.week[0])
sj_train["week_sin"] = np.sin(sj_train['week']/(52 * 2 * np.pi))
sj_train["week_cos"] = np.cos(sj_train['week']/(52 * 2 * np.pi))

sj_train.drop(['month','week'],axis=1, inplace = True)

sj_train = sj_train.dropna()


from sklearn.metrics import mean_squared_error, mean_absolute_error

sj_train_exog = sj_train.drop(['total_cases', 'week_start_date',], axis = 1)



model = SARIMAX(sj_train['total_cases'], order=(2, 0, 2), seasonal_order=(1, 0, 1, 52), exog=sj_train_exog)
model_fitted = model.fit()
predictions = model_fitted.predict(start = len(sj_train), end = len(sj_train)+len(sjTest)-1, exog=,dynamic=False)
compare_df = pd.concat([sj_train_subtest['total_cases'], predictions], axis=1).rename(columns={'total_cases': 'actual', 0:'predicted'})
compare_df.to_csv('tempdf1.csv')

plt.clf()
figs, axes = plt.subplots(nrows=1, ncols=1)
compare_df.actual.plot(ax=axes, label="actual")
compare_df.predicted_mean.plot(ax=axes, label="predicted")
plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
plt.savefig('idealSJ')

from sklearn.metrics import r2_score

compare_df.to_csv('compare.csv')
print(mean_squared_error(compare_df['actual'], compare_df['predicted_mean'], squared=False))
print(mean_absolute_error(compare_df['actual'], compare_df['predicted_mean']))
print(r2_score(compare_df['actual'], compare_df['predicted_mean']))

'''