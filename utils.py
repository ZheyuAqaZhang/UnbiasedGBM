import math, random
import numpy as np
from copy import deepcopy

def int_to_category(df):
    df = deepcopy(df)
    for c in df.columns:
        if 'int' in str(df[c].dtype):
            df[c] = df[c].astype('category')
    return df.rename(columns={c:f'Feat{i}' for i,c in enumerate(df.columns)})

def prepare_data(data, categorical_indicator, score_type):
    for c, t in zip(data.columns, categorical_indicator):
        if t:
            data[c] = data[c].astype('category').cat.codes.astype('int')
            if data[c].values.max() > 10 or 'xgb' in score_type:
                data[c] = data[c].astype('float')
            else:
                data[c] = data[c].astype('int' if '2.5' in score_type else 'category')
        else:
            data[c] = data[c].astype('float')
    return data
