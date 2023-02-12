
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

from datetime import datetime, date, timedelta

df = pd.DataFrame(np.nan, index=[datetime.strptime('1995-12-25', '%Y-%m-%d'), datetime.strptime('2000-12-25', '%Y-%m-%d'), datetime.strptime('2006-12-25', '%Y-%m-%d')], columns=sj_train.columns)
sj_train.set_index(pd.to_datetime(sj_train['week_start_date'], format='%Y-%m-%d'), inplace = True)
sj_train.index = pd.DatetimeIndex(sj_train.index)
sj_train = pd.concat([sj_train, df])


sj_train.index = pd.DatetimeIndex(sj_train.index).to_period('W') #period is one week
sj_train.sort_index(inplace = True)
sj_train.interpolate(option = 'spline')
sj_train.fillna(method='ffill', inplace=True)
#Extremely janky but works


sj_train['last_year_cases'] = sj_train['total_cases'].shift(52, axis = 0)
sj_train['1w'] = sj_train['total_cases'].shift(1, axis = 0)
sj_train['2w'] = sj_train['total_cases'].shift(2, axis = 0)
sj_train['3w'] = sj_train['total_cases'].shift(3, axis = 0)
sj_train['last3WeekAverage'] = (sj_train['1w']+sj_train['2w']+sj_train['3w'])/3

sj_train['diff1-2'] = sj_train['1w'] - sj_train['2w']
sj_train['diff2-3'] = sj_train['2w'] - sj_train['3w']
sj_train.drop(['1w','2w','3w'], axis = 1, inplace=True)

sj_train['month'] = int(sj_train.index.month[0])
sj_train['month_sin'] = np.sin(sj_train['month']/(12 * 2 * np.pi))
sj_train['month_cos'] = np.cos(sj_train['month']/(12 * 2 * np.pi))


sj_train['week'] = int(sj_train.index.week[0])
sj_train["week_sin"] = np.sin(sj_train['week']/(52 * 2 * np.pi))
sj_train["week_cos"] = np.cos(sj_train['week']/(52 * 2 * np.pi))

sj_train.drop(['month','week'],axis=1, inplace = True)




sj_train_subtrain = sj_train.head(800).dropna()
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)


from sklearn.metrics import mean_squared_error, mean_absolute_error

sj_train_subtrain_exog = sj_train_subtrain.drop(['total_cases', 'week_start_date',], axis = 1)
sj_train_subtest_exog = sj_train_subtest.drop(['total_cases', 'week_start_date'], axis = 1)

'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
sj_train_subtest_exog[sj_train_subtest_exog.columns] = scaler.fit_transform(sj_train_subtest_exog[sj_train_subtest_exog.columns])
sj_train_subtrain_exog[sj_train_subtrain_exog.columns] = scaler.fit_transform(sj_train_subtrain[sj_train_subtrain_exog.columns])
#minMax Scaling
'''
'''
def trainModel(exogTrain, exogTest):
    model = SARIMAX(sj_train_subtrain['total_cases'], order=(2, 0, 2), seasonal_order=(1, 0, 1, 52), exog=exogTrain)
    model_fitted = model.fit()
    predictions = model_fitted.predict(start = len(sj_train_subtrain), end = len(sj_train_subtrain)+len(sj_train_subtest)-1, exog=exogTest,dynamic=False)
    compare_df = pd.concat([sj_train_subtest['total_cases'], predictions], axis=1).rename(columns={'total_cases': 'actual', 0:'predicted'})
    from sklearn.metrics import r2_score
    return (mean_squared_error(compare_df['actual'], compare_df['predicted_mean'], squared=False), mean_absolute_error(compare_df['actual'], compare_df['predicted_mean']), r2_score(compare_df['actual'], compare_df['predicted_mean']))


data = {}
for i in sj_train_subtrain_exog.columns:
    data[i] = trainModel(sj_train_subtrain_exog.drop(i, axis = 1),sj_train_subtest_exog.drop(i, axis = 1))
print(data)

'''
print(sj_train_subtrain_exog.columns)

allVal = {'ndvi_ne': (12.526534189762597, 7.930668783366727, 0.8475072291870939),
'ndvi_nw': (12.594688998789422, 7.963501873428806, 0.8458433388994836),
'ndvi_se': (12.434630482908991, 7.832081014066145, 0.8497366152074146),
'ndvi_sw': (12.653752646312494, 7.984521361579888, 0.8443940924622046),
'precipitation_amt_mm': (12.47825847261825, 7.855132205652723, 0.8486803409547051),
'reanalysis_air_temp_k': (12.83605719025585, 8.011976008784707, 0.8398781178833161),
'reanalysis_avg_temp_k': (12.538976629198505, 7.876267334298976, 0.8472041406617838),
'reanalysis_dew_point_temp_k': (12.633683407176674, 7.923018868053975, 0.8448872925041593),
'reanalysis_max_air_temp_k': (12.511156442038045, 7.872246504569667, 0.8478814038709139),
'reanalysis_min_air_temp_k': (12.485692303839466, 7.861393767838609, 0.8484999920896433),
'reanalysis_precip_amt_kg_per_m2': (12.580358706066878, 7.839144288663621, 0.8461939395870138),
'reanalysis_relative_humidity_percent': (12.669413893378769, 7.933169599449213, 0.844008674676888),
'reanalysis_sat_precip_amt_mm': (12.488036276083118, 7.86088736634253, 0.8484431037509184),
'reanalysis_specific_humidity_g_per_kg': (12.576338151238948, 7.883132482470784, 0.8462922335861455),
'reanalysis_tdtr_k': (12.544694873142292, 7.890271028325478, 0.8470647475919014),
'station_avg_temp_c': (12.632473378743377, 7.951322379308599, 0.8449170038388614),
'station_diur_temp_rng_c': (12.767271214983978, 8.003022807687024, 0.8415896447683485),
'station_max_temp_c': (12.494013287321373, 7.864289474200387, 0.8482979930173606),
'station_min_temp_c': (12.46026972616098, 7.843412398839731, 0.8491163134780306),
'station_precip_mm': (12.511654811245052, 7.840189987123217, 0.8478692846500558),
'last_year_cases': (12.477858154708777, 7.85097337806621, 0.8486900498412502),
'1w': (12.465164901666034, 7.844149302520887, 0.8489977370383176),
'2w': (12.475086113267755, 7.855799466017098, 0.8487572714519837),
'3w': (12.476944779390742, 7.85682851560294, 0.8487122007126473),
'last3WeekAverage': (12.488340650869711, 7.862459885269924, 0.8484357157742816),
'diff1-2': (12.47466524790825, 7.851622459530401, 0.8487674760711796),
'diff2-3': (12.499575094511506, 7.867668732212622, 0.848162900298648),
'month_sin': (12.511119313267848, 7.87055069614882, 0.8478823067399722),
'month_cos': (12.480516428285517, 7.862566532717892, 0.8486255730565189),
'week_sin': (12.524899714770944, 7.881693635686491, 0.8475470214162315),
'week_cos': (12.480510034012344, 7.862560373725186, 0.8486257281669577),
'all': (12.51716911, 7.875885888, 0.847735157)}

names = (allVal.keys())
values = [allVal.get(name)[2] for name in names]
'''
plt.bar(range(len(allVal)), values, tick_label = names)
plt.show()
'''
a = [name for name in names if allVal.get(name)[2] > allVal.get('all')[2]]
b = [name for name in names if allVal.get(name)[1] < allVal.get('all')[1]]
c = [name for name in names if allVal.get(name)[1] < allVal.get('all')[0]]
print(list(set(a) & set(b) & set(c)))