import numpy as np
import pandas as pn
import utils, json, pickle
import matplotlib as plt
from sklearn.model_selection import train_test_split

path_to_csv = 'D:\OneDrive\Documents\Degree\Technion\Machine_Learning\hws\HW2'

df = pn.read_csv(r'{}\ElectionsData.csv'.format(path_to_csv), header=0)
# train, test = train_test_split(df, test_size=0.1)
# train, validation = train_test_split(train, test_size=0.25)

nd_dict = utils.get_features_probabilities_dict(df)

# description = df.describe()
# print(list(df.columns))

# Exploration:
col = 'Avg_monthly_expense_when_under_age_21'

utils.clean_data(df)
utils.feature_hist(df, col)










print('done')
