import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
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

#Let us determine the paramters experimentally, we know that differencing parameter is 0
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
mse_values = {}
'''
for p in range(1, 11):
    for q in range(1, 11):
        mse_values[(p,q)] = []
        model = ARIMA(sj_train_subtrain['total_cases'], order = (p, 0, q))
        model_fitted = model.fit()
        predictions = model_fitted.predict(start = 0, end = len(sj_train_subtrain)-1, dynamic=False)
        compare_df = pd.concat([sj_train_subtrain['total_cases'], predictions], axis=1).rename(
            columns={'total_cases': 'actual', 0: 'predicted'})
        compare_df.to_csv('tempdf1.csv')
        compare_df.dropna(inplace = True)
        mse = mean_squared_error(compare_df['actual'], compare_df['predicted_mean'])
        mse_values[(p,q)].append(mse)
mean_mse_values = {lag: np.mean(mse_values[lag]) for lag in mse_values}
best_lag = min(mean_mse_values, key=mean_mse_values.get)
print("best lag:" + str(best_lag))
'''
#Best lag is (4,5) when put on subtrain
#Best lag is (7,7)
#Best lag is (9, 0, 10)
'''
aic_values = {}
for p in range(1,10):
    for q in range(1,10):
        model = ARIMA(sj_train_subtrain['total_cases'], order = (p, 0, q))
        model_fitted = model.fit()
        aic_values[(p,q)] = (model_fitted.aic)
best_lag = min(aic_values, key=aic_values.get)
print("best lag:" + str(best_lag))
'''
#optiml is 3,8

model = ARIMA(sj_train_subtrain['total_cases'], order = (3,0,8))
model_fitted = model.fit()
predictions = model_fitted.predict(start = len(sj_train_subtrain), end = len(sj_train_subtrain)+len(sj_train_subtest)-1, dynamic=False)
compare_df = pd.concat([sj_train_subtest['total_cases'], predictions], axis=1).rename(columns={'total_cases': 'actual', 0:'predicted'})

plt.clf()
figs, axes = plt.subplots(nrows=1, ncols=1)
compare_df.actual.plot(ax=axes, label="actual")
compare_df.predicted_mean.plot(ax=axes, label="predicted")
plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
plt.savefig('ARIMA(3,0,8).png')
from sklearn.metrics import r2_score

compare_df.dropna(inplace = True)
print(mean_squared_error(compare_df['actual'], compare_df['predicted_mean'], squared=False))
print(mean_absolute_error(compare_df['actual'], compare_df['predicted_mean']))
print(r2_score(compare_df['actual'], compare_df['predicted_mean']))
