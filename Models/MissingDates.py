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


from datetime import datetime, date, timedelta

df = pd.DataFrame(np.nan, index=[datetime.strptime('2000-12-31', '%Y-%m-%d'), datetime.strptime('2006-12-31', '%Y-%m-%d')], columns=sj_train.columns)
iq_train.set_index(pd.to_datetime(iq_train['week_start_date'], format='%Y-%m-%d'), inplace = True)
iq_train.index = pd.DatetimeIndex(iq_train.index)
iq_train = pd.concat([iq_train, df])

iq_train.index = pd.DatetimeIndex(iq_train.index).to_period('W') #period is one week
iq_train.sort_index(inplace = True)
iq_train.fillna(method='ffill', inplace=True)
iq_train.to_csv('iq_train.csv')
#Extremely janky but works
'''
for i in pd.date_range(start='6-26-2000', end = '6-27-2010', freq='W'):
    if i not in iq_train.index:
        print(i)


2000-12-31 00:00:00
2006-12-31 00:00:00



iq_train['last_year_cases'] = sj_train['total_cases'].shift(52, axis = 0)
iq_train_subtrain = iq_train.head(400)
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)
iq_train_subtrain.dropna(inplace = True)



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

compare_df = pd.concat([sj_train_subtest['total_cases'], predictions], axis=1).rename(columns={'total_cases': 'actual', 0:'predicted'})
compare_df.to_csv('compare.csv')

plt.clf()
figs, axes = plt.subplots(nrows=1, ncols=1)
compare_df.actual.plot(ax=axes, label="actual")
compare_df.predicted.plot(ax=axes, label="predicted")
plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
#plt.savefig('ARX('+ str(best_lag) +')[last_year_cases, reanalysis_specific_humidity_g_per_kg, reanalysis_dew_point_temp_k, station_avg_temp_c, station_min_temp_c].png')
plt.show()
from sklearn.metrics import r2_score

'''