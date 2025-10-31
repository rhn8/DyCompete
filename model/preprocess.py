"""Some functions for preprocessing"""

import pandas as pd

def conditional_fill(group):

    transformed = group.copy()
    if pd.isna(transformed.iloc[0]):
        closest = transformed.bfill().iloc[0]
        transformed.iloc[0] = closest
    return transformed.ffill()



def process_patient(patient_data):

    patient_data_unique = patient_data.groupby(level='times').mean()
    new_index = pd.Index(range(0, 24), name='times')

    padded_data = patient_data_unique.reindex(new_index)

    return padded_data.ffill()

def process(timeseries):
    for col in timeseries:
        timeseries[col] = timeseries.groupby('id')[col].transform(conditional_fill)

    processed_df = timeseries.groupby('id').apply(process_patient)

    return processed_df

def transform(timeseries):

  static = timeseries.reset_index().groupby('id').first()

  return static.drop_duplicates()