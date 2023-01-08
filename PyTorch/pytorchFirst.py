import torch

def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg',
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

sj_train_subtrain = sj_train.head(800)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

model = torch.nn.LSTM(input_size=1, hidden_size=10, num_layers=2)


loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

forecaster = Model(model, loss_fn, optimizer)



forecaster.fit(train_data, epochs=100)
