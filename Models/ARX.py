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

sj_train, iq_train = preprocess_data('../CSV/dengue_features_train.csv',
                                     labels_path="../CSV/dengue_labels_train.csv")

sj_train.set_index(pd.to_datetime(sj_train['week_start_date'], format='%Y-%m-%d'), inplace = True)
sj_train.index = pd.DatetimeIndex(sj_train.index)
sj_train.index = pd.DatetimeIndex(sj_train.index).to_period('W')

sj_train['last_year_cases'] = sj_train['total_cases'].shift(52, axis = 0)
sj_train_subtrain = sj_train.head(800)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)
sj_train_subtrain.dropna(inplace = True)
sj_train_subtrain.to_csv('subtrainData.csv')
sj_train_subtest.to_csv('subtestData.csv')

#NEED TO FIND OPTIMAL LAG
from sklearn.metrics import mean_squared_error, mean_absolute_error

aic_values = {}


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(sj_train_subtrain['total_cases'])
plt.show()
plot_pacf(sj_train_subtrain['total_cases'])
plt.show()
'''
for lag in range(1, 10):
    model = AutoReg(sj_train_subtrain['total_cases'], exog = sj_train_subtrain[['last_year_cases', 'reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']], lags=lag)
    model_fitted = model.fit()
    predictions = model_fitted.predict(start = 0, end = len(sj_train_subtrain)-1, dynamic=False)
    aic_values[lag] = (model_fitted.aic)
best_lag = min(aic_values, key=aic_values.get)
print("best lag:" + str(best_lag))
print(min(aic_values))
#Best lag is 19 when put on subtrain
#Beat lag keeps going up
'''

'''
FAKE
'''
'''
mse_values = {}
for lag in range(1, 30):
    model = AutoReg(sj_train_subtrain['total_cases'],
                    exog=sj_train_subtrain[['last_year_cases', 'reanalysis_specific_humidity_g_per_kg',
                                            'reanalysis_dew_point_temp_k',
                                            'station_avg_temp_c',
                                            'station_min_temp_c']], lags=lag)
    model_fitted = model.fit()
    predictions = model_fitted.predict(start = len(sj_train_subtrain), end = len(sj_train_subtrain)+len(sj_train_subtest)-1, exog_oos=sj_train_subtest[['last_year_cases', 'reanalysis_specific_humidity_g_per_kg',
                                            'reanalysis_dew_point_temp_k',
                                            'station_avg_temp_c',
                                            'station_min_temp_c']], dynamic=False)
    compare_df = pd.concat([sj_train_subtest['total_cases'], predictions], axis=1).rename(
        columns={'total_cases': 'actual', 0: 'predicted'})
    compare_df.dropna(inplace = True)
    mse = mean_absolute_error(compare_df['actual'], compare_df['predicted'])
    mse_values[lag] = mse
best_lag = min(mse_values, key=mse_values.get)
print("best lag:" + str(best_lag))
'''

'''
mse_values = {}
for lag in range(1, 20):
    model = AutoReg(sj_train_subtrain['total_cases'],
                    exog=sj_train_subtrain[['last_year_cases', 'reanalysis_specific_humidity_g_per_kg',
                                            'reanalysis_dew_point_temp_k',
                                            'station_avg_temp_c',
                                            'station_min_temp_c']], lags=lag)
    model_fitted = model.fit()
    predictions = model_fitted.predict(start = 0, end = len(sj_train_subtrain)-1, dynamic=False)
    compare_df = pd.concat([sj_train_subtrain['total_cases'], predictions], axis=1).rename(
        columns={'total_cases': 'actual', 0: 'predicted'})
    compare_df.dropna(inplace = True)
    mse = mean_absolute_error(compare_df['actual'], compare_df['predicted'])
    mse_values[lag] = mse
best_lag = min(mse_values, key=mse_values.get)
print("best lag:" + str(best_lag))
#Best lag is 10 when put on subtrain
#Best lag is 283
'''
'''
best_lag = 18



#MAKE MODEL
arx_model = AutoReg(sj_train_subtrain['total_cases'], exog = sj_train_subtrain[['last_year_cases', 'reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']], lags = best_lag)
arx_results = arx_model.fit()

print(len(sj_train_subtrain))
print(len(sj_train_subtrain)+len(sj_train_subtest))
predictions = arx_results.predict(start = len(sj_train_subtrain), end = len(sj_train_subtrain)+len(sj_train_subtest)-1, exog_oos=sj_train_subtest[['last_year_cases', 'reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']])

predictions.to_csv('tempdf2.csv')
compare_df = pd.concat([sj_train_subtest['total_cases'], predictions], axis=1).rename(columns={'total_cases': 'actual', 0:'predicted'})

plt.clf()
figs, axes = plt.subplots(nrows=1, ncols=1)
compare_df.actual.plot(ax=axes, label="actual")
compare_df.predicted.plot(ax=axes, label="predicted")
compare_df.to_csv('tempdf1.csv')
plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
#plt.savefig('ARX('+ str(best_lag) +')[last_year_cases, reanalysis_specific_humidity_g_per_kg, reanalysis_dew_point_temp_k, station_avg_temp_c, station_min_temp_c].png')
from sklearn.metrics import r2_score

compare_df.dropna(inplace = True)
print(mean_squared_error(compare_df['actual'], compare_df['predicted'], squared=False))
print(mean_absolute_error(compare_df['actual'], compare_df['predicted']))
print(r2_score(compare_df['actual'], compare_df['predicted']))
'''