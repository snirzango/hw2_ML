import pandas as pn
from sklearn.model_selection import train_test_split

ElectionData = pn.read_csv(r'ElectionsData.csv', header=0)
label_name = 'Vote'

df_train, df_test = train_test_split(ElectionData, test_size=0.1)
df_train, df_validation = train_test_split(df_train, test_size=0.25)

features_info_dict = None

all_features = list(df_train.columns)
nominal_features_to_split = ['Most_Important_Issue', 'Will_vote_only_large_party', 'Main_transportation', 'Occupation']
binary_features_and_values = {'Looking_at_poles_results': {'Yes': 0, 'No': 1},
                                      'Financial_agenda_matters': {'Yes': 0, 'No': 1},
                                      'Married': {'Yes': 0, 'No': 1},
                                      'Gender': {'Female': 0, 'Male': 1},
                                      'Voting_Time': {'By_16:00': 0, 'After_16:00': 1},
                                      'Age_group': {'Below_30': 0, '30-45': 1, '45_and_up': 2}}

numeric_columns = [elem for elem in all_features if elem not in nominal_features_to_split and elem not in binary_features_and_values.keys()]
numeric_and_bool = [elem for elem in all_features if elem not in nominal_features_to_split]
