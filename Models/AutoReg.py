import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
#
# Load AutoReg class from statsmodels.tsa.ar_model module
#
from statsmodels.tsa.ar_model import AutoReg

def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    # select features we want
    features = ['week_start_date', 'reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']
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

sj_train, iq_train = preprocess_data('/dengue_features_train.csv',
                                     labels_path="/CSV/dengue_labels_train.csv")

sj_train['date'] = pd.to_datetime(sj_train['week_start_date'])
sj_train.set_index('date', inplace = True)
sj_train.index = pd.DatetimeIndex(sj_train.index).to_period('W') #frequency is weekly

sj_train_subtrain = sj_train.head(800)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

#plot_pacf(sj_train_subtrain, lags = 20)
#plt.show()
#Peak is at 1, so lag is only 1

#Find optimal Lag using function
from statsmodels.tsa.ar_model import ar_select_order
mod = ar_select_order(sj_train_subtrain['total_cases'], maxlag=20)
print(mod.ar_lags)
#optimal lag is first 6

#Let us figure out the optimal lag through rigorous testing
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse_values = {}

sj_train_subtrain.to_csv('tempdf1.csv')
'''
for lag in range(1, 11):
    mse_values[lag] = []
    model = AutoReg(sj_train_subtrain['total_cases'], lags=lag)
    model_fitted = model.fit()
    predictions = model_fitted.predict(start = 0, end = len(sj_train_subtrain)-1, dynamic=False)
    compare_df = pd.concat([sj_train_subtrain['total_cases'], predictions], axis=1).rename(
        columns={'total_cases': 'actual', 0: 'predicted'})
    compare_df.dropna(inplace = True)
    mse = mean_squared_error(compare_df['actual'], compare_df['predicted'])
    mse_values[lag].append(mse)
mean_mse_values = {lag: np.mean(mse_values[lag]) for lag in mse_values}
best_lag = min(mean_mse_values, key=mean_mse_values.get)
print("best lag:" + str(best_lag))
#Best lag is 9 when put on subtest
#Best lag is 10 when put on subtrain
'''
aic_values = {}
for lag in range(1,10):
    model = AutoReg(sj_train_subtrain['total_cases'], lags=lag)
    model_fitted = model.fit()
    aic_values[lag] = (model_fitted.aic)
best_lag = min(aic_values, key=aic_values.get)
print("best lag:" + str(best_lag))


model = AutoReg(sj_train_subtrain['total_cases'], lags = best_lag)
model_fitted = model.fit()
predictions = model_fitted.predict(start = len(sj_train_subtrain), end = len(sj_train_subtrain)+len(sj_train_subtest)-1, dynamic=False)
compare_df = pd.concat([sj_train_subtest['total_cases'], predictions], axis=1).rename(columns={'total_cases': 'actual', 0:'predicted'})
compare_df.to_csv('tempdf1.csv')

plt.clf()
figs, axes = plt.subplots(nrows=1, ncols=1)
compare_df.actual.plot(ax=axes, label="actual")
compare_df.predicted.plot(ax=axes, label="predicted")
plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
plt.savefig('AR(+' + str(best_lag)+ ').png')
from sklearn.metrics import r2_score

compare_df.dropna(inplace = True)
print(mean_squared_error(compare_df['actual'], compare_df['predicted'], squared=False))
print(mean_absolute_error(compare_df['actual'], compare_df['predicted']))
print(r2_score(compare_df['actual'], compare_df['predicted']))