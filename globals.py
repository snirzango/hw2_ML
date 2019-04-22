import pandas as pn
from sklearn.model_selection import train_test_split

ElectionData = pn.read_csv(r'ElectionsData.csv', header=0)
label_name = 'Vote'

df_train, df_test = train_test_split(ElectionData, test_size=0.1)
df_train, df_validation = train_test_split(df_train, test_size=0.25)
