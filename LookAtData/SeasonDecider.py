
from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
def decideSeason(week):
    if week >= 15 and week < 27:
        return "spring"
    elif(week>= 27 and week < 39):
        return "summer"
    elif(week>= 39 and week<51):
        return "autumn"
    elif(week >= 51 or week < 15):
        return "winter"
def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = data_path.set_index(['city','year','weekofyear'])
    #print(df.head())

    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_min_temp_c']
    df = df[features]

    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    #if labels_path:
    #labels = labels_path.set_index(['city','year','weekofyear'])
    df = df.join(labels_path.set_index(['city','year','weekofyear']))

    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']

    #just to be safe remove duplicates
    sj = sj[~sj.index.duplicated(keep='first')].copy()
    iq = iq[~iq.index.duplicated(keep='first')].copy()

    return sj, iq

def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_min_temp_c + " \
                    "station_avg_temp_c"

    grid = 10 ** np.arange(-8, -3, dtype=np.float64)

    best_alpha = []
    best_score = 1000

    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    #print('best alpha = ', best_alpha)
    #print('best score = ', best_score)

    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test]).astype(float)
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model

train_features = pd.read_csv('/dengue_features_train.csv')
train_labels = pd.read_csv('/dengue_labels_train.csv')

train_features['season'] = train_features['weekofyear'].apply(decideSeason)
train_labels['season'] = train_labels['weekofyear'].apply(decideSeason)

train_features_summer = train_features[train_features['season']=="summer"]
train_features_autumn = train_features[train_features['season']=="autumn"]
train_features_winter = train_features[train_features['season']=="winter"]
train_features_spring = train_features[train_features['season']=="spring"]
train_labels_summer = train_labels[train_labels['season']=="summer"]
train_labels_autumn = train_labels[train_labels['season']=="autumn"]
train_labels_winter = train_labels[train_labels['season']=="winter"]
train_labels_spring = train_labels[train_labels['season']=="spring"]

train_summer_sj, train_summer_iq = preprocess_data(train_features_summer, train_labels_summer)
train_autumn_sj, train_autumn_iq = preprocess_data(train_features_autumn, train_labels_autumn)
train_winter_sj, train_winter_iq = preprocess_data(train_features_winter, train_labels_winter)
train_spring_sj, train_spring_iq = preprocess_data(train_features_spring, train_labels_spring)

train_summer_sj_subtrain = train_summer_sj.head(int(train_summer_sj.shape[0] * 4/5))
train_summer_sj_subtest = train_summer_sj.tail(int(train_summer_sj.shape[0]*1/5))
train_summer_iq_subtrain = train_summer_iq.head(int(train_summer_sj.shape[0]*4/5))
train_summer_iq_subtest = train_summer_iq.tail(int(train_summer_sj.shape[0]*1/5))
train_autumn_sj_subtrain = train_autumn_sj.head(int(train_summer_sj.shape[0]*4/5))
train_autumn_sj_subtest = train_autumn_sj.tail(int(train_summer_sj.shape[0]*1/5))
train_autumn_iq_subtrain = train_autumn_iq.head(int(train_summer_sj.shape[0]*4/5))
train_autumn_iq_subtest = train_autumn_iq.tail(int(train_summer_sj.shape[0]*1/5))
train_spring_sj_subtrain = train_spring_sj.head(int(train_summer_sj.shape[0]*4/5))
train_spring_sj_subtest = train_spring_sj.tail(int(train_summer_sj.shape[0]*1/5))
train_spring_iq_subtrain = train_spring_iq.head(int(train_summer_sj.shape[0]*4/5))
train_spring_iq_subtest = train_spring_iq.tail(int(train_summer_sj.shape[0]*1/5))
train_winter_sj_subtrain = train_winter_sj.head(int(train_summer_sj.shape[0]*4/5))
train_winter_sj_subtest = train_winter_sj.tail(int(train_summer_sj.shape[0]*1/5))
train_winter_iq_subtrain = train_winter_iq.head(int(train_summer_sj.shape[0]*4/5))
train_winter_iq_subtest = train_winter_iq.tail(int(train_summer_sj.shape[0]*1/5))



#print(train_summer_sj_subtrain.keys())
#print(train_summer_sj_subtrain.drop(columns = 'season').keys())
tempdf1=train_summer_sj_subtrain.drop(columns = ['season']).copy()
tempdf2=train_summer_sj_subtest.drop(columns = ['season']).copy()
tempdf1.to_csv('tempdf1.csv')
tempdf2.to_csv('tempdf2.csv')

summer_sj_best_model = get_best_model(tempdf1, tempdf2)
summer_iq_best_model = get_best_model(train_summer_iq_subtrain.drop(columns = 'season'), train_summer_iq_subtest.drop(columns = 'season'))
autumn_sj_best_model = get_best_model(train_autumn_sj_subtrain.drop(columns = 'season'), train_autumn_sj_subtest.drop(columns = 'season'))
autumn_iq_best_model = get_best_model(train_autumn_iq_subtrain.drop(columns = 'season'), train_autumn_iq_subtest.drop(columns = 'season'))
winter_sj_best_model = get_best_model(train_winter_sj_subtrain.drop(columns = 'season'), train_winter_sj_subtest.drop(columns = 'season'))
winter_iq_best_model = get_best_model(train_winter_iq_subtrain.drop(columns = 'season'), train_winter_iq_subtest.drop(columns = 'season'))
spring_sj_best_model = get_best_model(train_spring_sj_subtrain.drop(columns = 'season'), train_spring_sj_subtest.drop(columns = 'season'))
spring_iq_best_model = get_best_model(train_spring_iq_subtrain.drop(columns = 'season'), train_spring_iq_subtest.drop(columns = 'season'))

figs, axes = plt.subplots(nrows=2, ncols=4)
axe = axes.ravel()
train_summer_sj.to_csv('tempdf1.csv')
train_summer_iq.to_csv('tempdf2.csv')
#print(train_summer_sj.index.is_unique)
#print(train_summer_iq.index.is_unique)
#print(summer_iq_best_model.fittedvalues.index)


#print(train_summer_sj.shape[0] == (len(summer_sj_best_model.fittedvalues)))
#print(train_summer_iq.shape[0] == len(summer_iq_best_model.fittedvalues))



#iq_fitted.to_csv('iq_fitted.csv')
#sj_fitted.to_csv('sj_fitted.csv')

train_summer_iq['fitted'] = summer_iq_best_model.fittedvalues.drop_duplicates(keep='first')
train_summer_iq.fitted.plot(ax=axe[4], label="Predictions")
train_summer_iq.total_cases.plot(ax=axe[4], label="Actual")

print(train_summer_iq.head)
train_summer_iq.to_csv('tempdf1.csv')


train_autumn_iq['fitted'] = autumn_iq_best_model.fittedvalues.drop_duplicates(keep='first')
train_autumn_iq.fitted.plot(ax=axe[5], label="Predictions")
train_autumn_iq.total_cases.plot(ax=axe[5], label="Actual")


train_winter_iq['fitted'] = winter_iq_best_model.fittedvalues.drop_duplicates(keep='first')
train_winter_iq.fitted.plot(ax=axe[6], label="Predictions")
train_winter_iq.total_cases.plot(ax=axe[6], label="Actual")

train_spring_iq['fitted'] = spring_iq_best_model.fittedvalues.drop_duplicates(keep='first')
train_spring_iq.fitted.plot(ax=axe[7], label="Predictions")
train_spring_iq.total_cases.plot(ax=axe[7], label="Actual")


train_summer_sj['fitted'] = summer_sj_best_model.fittedvalues.drop_duplicates(keep='first')
train_summer_sj.fitted.plot(ax=axe[0], label="Predictions")
train_summer_sj.total_cases.plot(ax=axe[0], label="Actual")

train_autumn_sj['fitted'] = autumn_sj_best_model.fittedvalues.drop_duplicates(keep='first')
train_autumn_sj.fitted.plot(ax=axe[1], label="Predictions")
train_autumn_sj.total_cases.plot(ax=axe[1], label="Actual")

train_winter_sj['fitted'] = winter_sj_best_model.fittedvalues.drop_duplicates(keep='first')
train_winter_sj.fitted.plot(ax=axe[2], label="Predictions")
train_winter_sj.total_cases.plot(ax=axe[2], label="Actual")

train_spring_sj['fitted'] = spring_sj_best_model.fittedvalues.drop_duplicates(keep='first')
train_spring_sj.fitted.plot(ax=axe[3], label="Predictions")
train_spring_sj.total_cases.plot(ax=axe[3], label="Actual")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
#plt.show()

frames_iq = [train_summer_iq,train_winter_iq,train_autumn_iq,train_spring_iq]
frames_sj = [train_summer_sj, train_winter_sj, train_autumn_sj, train_spring_sj]

sj_train = pd.concat(frames_sj)
sj_train.sort_index(ascending=True, inplace=True)
print(sj_train.head)
sj_train.to_csv('tempdf1.csv')
#sj_train.set_index(['year','weekofyear'])

iq_train = pd.concat(frames_iq)
iq_train.sort_index(ascending=True, inplace=True)
#iq_train.set_index(['year','weekofyear'])

figs, axes = plt.subplots(nrows=2, ncols=1)

sj_train.fitted.plot(ax=axes[0], label="Predictions")
sj_train.total_cases.plot(ax=axes[0], label="Actual")

iq_train.fitted.plot(ax=axes[1], label="Predictions")
iq_train.total_cases.plot(ax=axes[1], label="Actual")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
#plt.show()


sj_train.reset_index(inplace=True)
iq_train.reset_index(inplace=True)
sj_train.dropna(inplace=True)
iq_train.dropna(inplace=True)
print(sj_train.head())
sj_MAPE = mean_absolute_error(sj_train.total_cases, sj_train.fitted)
iq_MAPE = mean_absolute_error(iq_train.total_cases, iq_train.fitted)
sj_rmse = mean_squared_error(sj_train.total_cases, sj_train.fitted, squared=False)
iq_rmse = mean_squared_error(iq_train.total_cases, iq_train.fitted, squared=False)

print(sj_MAPE,iq_MAPE,sj_rmse,iq_rmse)