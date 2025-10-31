import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch # For building the networks
import torchtuples as tt # Some useful functions

from pycox.models import LogisticHazard
from pycox.evaluation import EvalSurv

import seaborn as sn
sn.set_theme(style="white", palette="rocket_r")

import random

from preprocess import process, transform
from losses import Loss
from model_utils import DyCompete




def main():

    seed = 1234 #(1024, 85858, 3673, 32)

    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(123)

    df = pd.read_csv('../data/pbc2_cleaned.csv', index_col=['id', 'times'])
    ##################### Initial data preprocessing ##################

    df_processed = process(df)

    ids = df_processed.index.get_level_values('id').unique()

    train_ids, temp_ids = train_test_split(
        ids,
        test_size=0.3, # 40% of the data will be for validation and testing
        random_state=42
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5, # 50% of the 40% temp set
        random_state=42
    )

    timeseries_train = df_processed.loc[train_ids]
    timeseries_val = df_processed.loc[val_ids]
    timeseries_test = df_processed.loc[test_ids]

    # Mean imputation using training set 
    timeseries_train['serChol'] = timeseries_train['serChol'].fillna(timeseries_train['serChol'].mean())
    timeseries_val['serChol'] = timeseries_val['serChol'].fillna(timeseries_train['serChol'].mean())
    timeseries_test['serChol'] = timeseries_test['serChol'].fillna(timeseries_train['serChol'].mean())


    cols_standardize = ['age', 'edema', 'serBilir', 'serChol', 'albumin', 'alkaline',
                        'SGOT', 'platelets', 'prothrombin', 'histologic']

    cols_leave = ['drug', 'sex', 'ascites', 'hepatomegaly', 'spiders']


    df_train = transform(timeseries_train)
    df_val = transform(timeseries_val)
    df_test = transform(timeseries_test)


    ##################### Final data processing ################## 


    num_durations = 10
    labtrans = LogisticHazard.label_transform(num_durations)

    all_events_train = (df_train['label'].values > 0).astype('int64')
    labtrans.fit(df_train['tte'].values, all_events_train)

    get_target = lambda df, cause_id: (df['tte'].values,(df['label'].values == cause_id).astype('int64'))

    # Get time to event data for individual competing events
    num_risks = df_train['label'].nunique()

    y_train_risks = {}
    y_val_risks = {}
    for cause_id in range(1, num_risks):
        y_train_risks[cause_id] = labtrans.transform(*get_target(df_train, cause_id))
        y_val_risks[cause_id] = labtrans.transform(*get_target(df_val, cause_id))

    y_train_surv = y_train_risks[1]
    y_val_surv = y_val_risks[1]

    for cause_id in range(2, num_risks):
        y_train_surv = (y_train_surv[0], y_train_surv[1] + cause_id*y_train_risks[cause_id][1])
        y_val_surv = (y_val_surv[0], y_val_surv[1] + cause_id*y_val_risks[cause_id][1])

    # Apply standardisation to continuous data
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]

    x_mapper = DataFrameMapper(standardize + leave)

    # Check data type is correct
    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')


    train = tt.tuplefy((x_train, y_train_surv[0]), (y_train_surv, x_train))
    val = tt.tuplefy((x_val, y_val_surv[0]), (y_val_surv, x_val))

    durations_test, events_test = get_target(df_test, cause_id=(1 or 2))

    ##################### Get Data Properties ################## 

    in_features = x_train.shape[1]
    encoded_features = 20 # use 20 latent factors
    out_features = labtrans.out_features # how many discrete time points to predict for (10 here)


    net = DyCompete(in_features, encoded_features, out_features, risks=2)


    loss = Loss([0.2, 0.5, 0 , 0.3, 0]
    )


    model = LogisticHazard(net, tt.optim.Adam(0.001), duration_index=labtrans.cuts, loss=loss) # wrapper


    metrics = dict(
        loss_surv = Loss(alpha=[1, 0, 0, 0, 0]),
        loss_ae = Loss(alpha=[0, 1, 0, 0, 0]),
        loss_kd = Loss(alpha=[0, 0, 1, 0, 0]),
        loss_competing = Loss(alpha=[0, 0, 0, 1, 0]),
        loss_ranking = Loss(alpha=[0, 0, 0, 0, 1])
    )
    callbacks = [tt.cb.EarlyStopping()]


    batch_size = 256
    epochs = 1000
    log = model.fit(*train, batch_size = batch_size, epochs = epochs, callbacks = callbacks, verbose = True, val_data=val, metrics=metrics)

    res = model.log.to_pandas()

    surv = model.interpolate(10).predict_surv_df(x_test)

    ##################### Evaluate model ################## 

    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')

    time_grid = np.linspace(durations_test.min(), durations_test.max(), 1000)


    print(f'IBS: {ev.integrated_brier_score(time_grid)}')


    print(f'Time dependent concordance: {ev.concordance_td('adj_antolini')}')

    print(f'nBLL: {ev.integrated_nbll(time_grid)}')

if __name__ == "__main__":
    main()