import gc
import time
import random
import warnings
import feather

import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import generate_segment_start_ids, compare_methods
from features import create_all_features_extended_denoised

#Configure the environment
pd.options.display.precision = 15
warnings.filterwarnings('ignore')
random.seed(1013)

# Load/compute the necessary features
compute_features = True 
train_data_format = 'feather'

## Training data
def load_train_data(file_format):
    """Load the training dataset."""
    print(f"Loading data from {file_format} file:", end="")
    if file_format.lower() == 'feather':
        train_df = feather.read_dataframe('../input/train.feather')
    else:
        train_df = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16,
                                                            'time_to_failure': np.float32})
        feather.write_dataframe(train_df, '../input/train.feather')
    print("Done")
    return train_df


train = load_train_data(train_data_format)

time_to_failure_delta = np.diff(train['time_to_failure'])
init_times = np.where(time_to_failure_delta > 5)[0].tolist()
print(f"There are {len(init_times)} quakes on the training set.")
init_times = [0] + init_times
d = {'start_idx': init_times, 'end_idx': init_times[1:] + [len(time_to_failure_delta)]}
quakes = pd.DataFrame(data=d)
quakes.insert(2, 'valid', True)


# ## Feature generation
# - Usual aggregations: mean, std, min and max;
# - Average difference between the consequitive values in absolute and percent values;
# - Absolute min and max vallues;
# - Aforementioned aggregations for first and last 10000 and 50000 values - I think these data should be useful;
# - Max value to min value and their differencem also count of values bigger that 500 (arbitrary threshold);
# - Quantile features from this kernel: https://www.kaggle.com/andrekos/basic-feature-benchmark-with-quantiles
# - Trend features from this kernel: https://www.kaggle.com/jsaguiar/baseline-with-abs-and-trend-features
# - Rolling features from this kernel: https://www.kaggle.com/wimwim/rolling-quantiles

fs = 4000000 #Sampling frequency of the raw signal

#Compute features for the training data
segment_size = 150000
segment_start_ids = generate_segment_start_ids('uniform_no_jump', segment_size, train)
X_tr = pd.DataFrame(index=range(len(segment_start_ids)), dtype=np.float64)
y_tr = pd.DataFrame(index=range(len(segment_start_ids)), dtype=np.float64, columns=['time_to_failure'])

for idx in tqdm(range(len(segment_start_ids))):
    seg_id = segment_start_ids[idx]
    seg = train.iloc[seg_id:seg_id + segment_size]
    create_all_features_extended_denoised(idx, seg, X_tr, fs)
    y_tr.loc[idx, 'time_to_failure'] = seg['time_to_failure'].values[-1]

# Sanity check
means_dict = {}
for col in X_tr.columns:
    if X_tr[col].isnull().any():
        print(col)
        mean_value = X_tr.loc[X_tr[col] != -np.inf, col].mean()
        X_tr.loc[X_tr[col] == -np.inf, col] = mean_value
        X_tr[col] = X_tr[col].fillna(mean_value)
        means_dict[col] = mean_value

#Compute features for the test data
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_tr.columns, dtype=np.float64, index=submission.index)
for i, seg_id in enumerate(tqdm(X_test.index)):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    create_all_features_extended_denoised(seg_id, seg, X_test, fs)

# Sanity check
for col in X_test.columns:
    if X_test[col].isnull().any():
        X_test.loc[X_test[col] == -np.inf, col] = means_dict[col]
        X_test[col] = X_test[col].fillna(means_dict[col])
        
X_tr.to_hdf('../tmp_results/Xv3_denoised_tr.hdf', 'data')
X_test.to_hdf('../tmp_results/Xv3_denoised_test.hdf', 'data')
y_tr.to_hdf('../tmp_results/yv3_denoised_tr.hdf', 'data')
