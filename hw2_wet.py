import numpy as np
import pandas as pn
import utils, json, pickle
import matplotlib as plt
from sklearn.model_selection import train_test_split

path_to_csv = 'D:\OneDrive\Documents\Degree\Technion\Machine_Learning\hws\HW2'

df = pn.read_csv(r'{}\ElectionsData.csv'.format(path_to_csv), header=0)
# train, test = train_test_split(df, test_size=0.1)
# train, validation = train_test_split(train, test_size=0.25)

# Clean data:
# features_info_dict = utils.get_features_info_dict(train)
# for df in [train, validation, test]:
#     utils.clean_data(df, features_info_dict)
utils.clean_data(df)


# Exploration:
col = 'Garden_sqr_meter_per_person_in_residancy_area'

utils.feature_hist(df, col, bins=10)

# print(df['Financial_balance_score_(0-1)'].value_counts())









print('done')
