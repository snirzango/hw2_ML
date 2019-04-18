import numpy as np
import pandas as pn
import utils, json, pickle

df = pn.read_csv(r'ElectionsData.csv', header=0)
# train, test = train_test_split(df, test_size=0.1)
# train, validation = train_test_split(train, test_size=0.25)


nd_dict = utils.get_features_info_dict(df)
df = utils.clean_data(df, nd_dict)

utils.feature_selection_RFE_test(df)
