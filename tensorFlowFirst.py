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
import seaborn as sns
import tensorflow as tf


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

sj_train, iq_train = preprocess_data('CSV/dengue_features_train.csv',
                                     labels_path="CSV/dengue_labels_train.csv")


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
#this is the first one with correct dates



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

#Split data
n = len(sj_train)
sj_train_subtrain = sj_train[0:int(n*0.7)]
sj_train_subvalidate = sj_train[int(n*0.7):int(n*0.9)]
sj_train_subtest = sj_train[int(n*0.9):]

#normalize features
sj_train_subtrain_mean = sj_train_subtrain.mean()
sj_train_subtrain_std = sj_train_subtrain.std()

sj_train_subtrain = (sj_train_subtrain-sj_train_subtrain_mean)/sj_train_subtrain_std
sj_train_subvalidate = (sj_train_subvalidate-sj_train_subtrain_mean)/sj_train_subtrain_std
sj_train_subtest = (sj_train_subtest-sj_train_subtrain_mean)/sj_train_subtrain_std

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=sj_train_subtrain, val_df=sj_train_subvalidate, test_df=sj_train_subtest,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])



w1 = WindowGenerator(input_width = 845, label_width= 94, shift = 1, label_columns=['total_cases'])

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window# Stack three slices, the length of the total window.

