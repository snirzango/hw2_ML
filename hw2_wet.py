import numpy as np
import pandas as pn
import utils
from sklearn.model_selection import train_test_split
import json

df = pn.read_csv(r'D:\OneDrive\Documents\Degree\Technion\Machine_Learning\hws\HW2\ElectionsData.csv', header=0)
# train, test = train_test_split(df, test_size=0.1)
# train, validation = train_test_split(train, test_size=0.25)

tmp_dict = utils.get_features_probabilities_dict(df)

# description = df.describe()
# utils.split_nominal_feature_to_bool_features(df, 'Most_Important_Issue')
#
# print(list(df.columns))
# print(df.Most_Important_Issue_Foreign_Affairs.count())


json.dump(tmp_dict, open('prob_dict.json', 'w'))
print('done')
