import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
#
# Load AutoReg class from statsmodels.tsa.ar_model module
#
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX


def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    # select features we want
    df = df

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

from datetime import datetime, date, timedelta

df = pd.DataFrame(np.nan, index=[datetime.strptime('1995-12-25', '%Y-%m-%d'), datetime.strptime('2000-12-25', '%Y-%m-%d'), datetime.strptime('2006-12-25', '%Y-%m-%d')], columns=sj_train.columns)
sj_train.set_index(pd.to_datetime(sj_train['week_start_date'], format='%Y-%m-%d'), inplace = True)
sj_train.index = pd.DatetimeIndex(sj_train.index)
sj_train = pd.concat([sj_train, df])
sj_train.index = pd.DatetimeIndex(sj_train.index).to_period('W') #period is one week
sj_train.sort_index(inplace = True)
sj_train.interpolate(option = 'spline')
sj_train.fillna(method='ffill', inplace=True)



sj_train['last_year_cases'] = sj_train['total_cases'].shift(52, axis = 0)
sj_train_subtrain = sj_train.head(800)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)
sj_train_subtrain.dropna(inplace = True)

sj_train_subtrain_exog = sj_train_subtrain['last_year_cases']
sj_train_subtest_exog = sj_train_subtest['last_year_cases']

#FIND OPTIMAL paramters
#No need use from SARIMA

from sklearn.metrics import mean_squared_error, mean_absolute_error

model = SARIMAX(sj_train_subtrain['total_cases'], order=(2, 0, 2), seasonal_order=(1, 0, 1, 52), exog=sj_train_subtrain_exog, axis = 1, dynamic=False)
model_fitted = model.fit()
predictions = model_fitted.predict(start = len(sj_train_subtrain), end = len(sj_train_subtrain)+len(sj_train_subtest)-1, exog=sj_train_subtest_exog,dynamic=False)
compare_df = pd.concat([sj_train_subtest['total_cases'], predictions], axis=1).rename(columns={'total_cases': 'actual', 0:'predicted'})

plt.clf()
figs, axes = plt.subplots(nrows=1, ncols=1)
compare_df.actual.plot(ax=axes, label="actual")
compare_df.predicted_mean.plot(ax=axes, label="predicted")
plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
#plt.savefig('SARIMAX(2,0,2)(1,0,1,52)[reanalysis_specific_humidity_g_per_kg, reanalysis_dew_point_temp_k, station_avg_temp_c, station_min_temp_c]')
plt.show()
from sklearn.metrics import r2_score

compare_df.dropna(inplace = True)
print(mean_squared_error(compare_df['actual'], compare_df['predicted_mean'], squared=False))
print(mean_absolute_error(compare_df['actual'], compare_df['predicted_mean']))
print(r2_score(compare_df['actual'], compare_df['predicted_mean']))