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

from datetime import datetime, date, timedelta

df = pd.DataFrame(np.nan, index=[datetime.strptime('2000-12-31', '%Y-%m-%d'), datetime.strptime('2006-12-31', '%Y-%m-%d')], columns=iq_train.columns)
iq_train.set_index(pd.to_datetime(iq_train['week_start_date'], format='%Y-%m-%d'), inplace = True)
iq_train.index = pd.DatetimeIndex(iq_train.index)
iq_train = pd.concat([iq_train, df])

iq_train.index = pd.DatetimeIndex(iq_train.index).to_period('W') #period is one week
iq_train.sort_index(inplace = True)
iq_train.interpolate(option = 'spline', inplace=True)
iq_train.fillna(method='ffill', inplace=True)

#Extremely janky but works

'''
iq_train['last_year_cases'] = iq_train['total_cases'].shift(52, axis = 0)
iq_train['1w'] = iq_train['total_cases'].shift(1, axis = 0)
iq_train['2w'] = iq_train['total_cases'].shift(2, axis = 0)
iq_train['3w'] = iq_train['total_cases'].shift(3, axis = 0)
iq_train['last3WeekAverage'] = (iq_train['1w']+iq_train['2w']+iq_train['3w'])/3
iq_train.drop(['1w','2w','3w'], axis = 1)
iq_train['diff1-2'] = iq_train['1w'] - iq_train['2w']
iq_train['diff2-3'] = iq_train['2w'] - iq_train['3w']
'''

iq_train['4yearsAgo'] = iq_train['total_cases'].shift(208, axis = 0)

iq_train['month'] = int(iq_train.index.month[0])
iq_train['month_sin'] = np.sin(iq_train['month']/(12 * 2 * np.pi))
iq_train['month_cos'] = np.cos(iq_train['month']/(12 * 2 * np.pi))


iq_train['week'] = int(iq_train.index.week[0])
iq_train["week_sin"] = np.sin(iq_train['week']/(52 * 2 * np.pi))
iq_train["week_cos"] = np.cos(iq_train['week']/(52 * 2 * np.pi))

iq_train.drop(['month','week'],axis=1, inplace = True)


iq_train_subtrain = iq_train.head(400).dropna()
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)


from sklearn.metrics import mean_squared_error, mean_absolute_error

iq_train_subtrain_exog = iq_train_subtrain.drop(['total_cases', 'week_start_date',], axis = 1)
iq_train_subtest_exog = iq_train_subtest.drop(['total_cases', 'week_start_date'], axis = 1)

'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
iq_train_subtest_exog[iq_train_subtest_exog.columns] = scaler.fit_transform(iq_train_subtest_exog[iq_train_subtest_exog.columns])
iq_train_subtrain_exog[iq_train_subtrain_exog.columns] = scaler.fit_transform(iq_train_subtrain[iq_train_subtrain_exog.columns])
#minMax Scaling
'''
'''
aic_values = {}
for a in range(1,4):
    for b in range(1,4):
        for c in range(1,4):
            for d in range(1,4):
                model = SARIMAX(iq_train_subtrain['total_cases'], order = (a, 0, b), seasonal_order=(c,0,d,52), exog=iq_train_subtrain_exog)
                model_fitted = model.fit(disp=True)
                aic_values[(a,b,c,d)] = (model_fitted.aic)
                print(str(a) + " "+str(b) +" "+ str(c) + " " + str(d) + " " + str(model_fitted.aic))
best_lag = min(aic_values, key=aic_values.get)
print("best lag:" + str(best_lag))

#optimal is 3,8

'''

model = SARIMAX(iq_train_subtrain['total_cases'], order=(1, 0, 2), seasonal_order=(2, 0, 1, 52), exog=iq_train_subtrain_exog)
model_fitted = model.fit()
predictions = model_fitted.predict(start = len(iq_train_subtrain), end = len(iq_train_subtrain)+len(iq_train_subtest)-1, exog=iq_train_subtest_exog,dynamic=False)
compare_df = pd.concat([iq_train_subtest['total_cases'], predictions], axis=1).rename(columns={'total_cases': 'actual', 0:'predicted'})


plt.clf()
figs, axes = plt.subplots(nrows=1, ncols=1)
compare_df.actual.plot(ax=axes, label="actual")
compare_df.predicted_mean.plot(ax=axes, label="predicted")
plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
plt.savefig('Iq[4 years ago, all weather]')

from sklearn.metrics import r2_score

compare_df.to_csv('compare.csv')
print(mean_squared_error(compare_df['actual'], compare_df['predicted_mean'], squared=False))
print(mean_absolute_error(compare_df['actual'], compare_df['predicted_mean']))
print(r2_score(compare_df['actual'], compare_df['predicted_mean']))

