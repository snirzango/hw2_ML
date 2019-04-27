from globals import *

import numpy as np
import pandas as pn
import category_encoders as ce
import matplotlib as plt

import warnings

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score


def drop_label_column(df=df_train):
    return df.drop(label_name, axis=1)


def get_label_column(df=df_train):
    return df[label_name]


def fill_missing_values(features_info_dict, df=df_train):
    features = list(df.columns)
    for feature in features:
        df[feature] = df[feature].apply(lambda v: choose_mean_value(features_info_dict, feature) if str(v) == 'nan' else v)
    return df


def get_features_info_dict(df=df_train, eliminate_nd_elements=True):
    features = list(df.columns)  # features[0] is labels
    features_dict = {}

    # Counting and will later normalize by total_feature_size
    for feature in features:
        is_nominal = df[feature].dtype == np.object
        values_init = np.array([]) if is_nominal else None
        probs_init = np.array([]) if is_nominal else None
        mean_init = None if is_nominal else -1

        feature_info_dict = {'is_nominal': is_nominal, 'values': values_init, 'probs': probs_init, 'mean': mean_init}

        if is_nominal:
            values_count = df[feature].value_counts()
            feature_total_not_missing_values = sum(values_count.values)
            feature_info_dict['values'] = values_count.keys().values
            feature_info_dict['probs'] = values_count.values / feature_total_not_missing_values
        else:
            feature_info_dict['mean'] = df[feature].mean()

        if eliminate_nd_elements:
            for key, value in feature_info_dict.items():
                if isinstance(value, np.ndarray):
                    feature_info_dict[key] = value.tolist()

        features_dict[feature] = feature_info_dict

    return features_dict


def choose_mean_value(features_info_dict, feature_name):
    if features_info_dict[feature_name]['is_nominal']:
        names = features_info_dict[feature_name]['values']
        probs = features_info_dict[feature_name]['probs']
        return np.random.choice(names, 1, p=probs)[0]
    else:
        return features_info_dict[feature_name]['mean']


def replace_negatives_with_mean(features, df=df_train):
    for feature in features:
        mean = df[feature][df[feature] >= 0].mean()
        df[feature] = df[feature].apply(lambda v: mean if v < 0 else v)


def feature_hist(feature, df=df_train, bins=100):
    df.hist(column=feature, bins=bins)
    plt.pyplot.show()


def find_outliers(df=df_train):
    ObjFeat = df.keys()[df.dtypes.map(lambda x: x == 'object')]

    for f in ObjFeat:
        df[f] = df[f].astype("str")
        df[f] = df[f].astype("category")
        df[f] = df[f].cat.rename_categories(range(df[f].nunique())).astype(int)
        df.loc[df[f].isnull(), f] = np.nan  # fix NaN conversion
    X_train = drop_label_column(df)
    clf = IsolationForest()
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    print(50 * '-')
    print(y_pred_train._len_())
    count = 0
    for pred in y_pred_train:

        if pred == -1:
            count += 1

    print(count)


def clean_data(df=df_train, features_info_dict=None, negative_to_mean=True, labels_to_unique_ints=True, nominal_to_bool_split=True,
               missing_values_fill=True, binary_to_numeric=True, normalization=True):
    ''' Main function to clean data '''

    # TODO: add "features to drop" section.

    if negative_to_mean:
        # Replacing negative numbers with mean for appropriate columns:
        features_to_apply = ['Avg_monthly_expense_when_under_age_21']
        replace_negatives_with_mean(df, features_to_apply)

    if labels_to_unique_ints:
        # Convert Vote (label) to unique numbers
        df[label_name] = df[label_name].astype("str")
        df[label_name] = df[label_name].astype("category")
        df[label_name] = df[label_name].cat.rename_categories(range(df[label_name].nunique())).astype(int)
        df.loc[df[label_name].isnull(), label_name] = np.nan  # fix NaN conversion

    if nominal_to_bool_split:
        # Nominal features splitting:
        nominal_features_to_split = ['Most_Important_Issue', 'Will_vote_only_large_party', 'Main_transportation',
                                     'Occupation']
        df = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True, cols=nominal_features_to_split).fit_transform(df)

    if features_info_dict is None:
        features_info_dict = get_features_info_dict(df)

    if missing_values_fill:
        #  Fill missing values by probability/mean:
        df = fill_missing_values(features_info_dict=features_info_dict, df=df)

    if binary_to_numeric:
        # Binary nominal (yes/no etc') to numeric (0/1):
        binary_features_and_values = {'Looking_at_poles_results': {'Yes': 0, 'No': 1},
                                      'Financial_agenda_matters': {'Yes': 0, 'No': 1},
                                      'Married': {'Yes': 0, 'No': 1},
                                      'Gender': {'Female': 0, 'Male': 1},
                                      'Voting_Time': {'By_16:00': 0, 'After_16:00': 1},
                                      'Age_group': {'Below_30': 0, '30-45': 1, '45_and_up': 2}}
        for feature, replacement_dict in binary_features_and_values.items():
            df[feature] = df[feature].map(replacement_dict)

    if normalization:
        #  Normalization phase:
        features_to_normalize_0_to_1 = ['Age_group', 'Avg_Residancy_Altitude', 'Avg_size_per_room',
                                        'Garden_sqr_meter_per_person_in_residancy_area', 'Num_of_kids_born_last_10_years',
                                        'Number_of_differnt_parties_voted_for', 'Number_of_valued_Kneset_members',
                                        'Phone_minutes_10_years']
        features_to_normalize_minus1_to_1 = ['AVG_lottary_expanses', 'Avg_monthly_expense_on_pets_or_plants',
                                             'Occupation_Satisfaction', 'Avg_environmental_importance',
                                             'Avg_Satisfaction_with_previous_vote', 'Avg_education_importance',
                                             'Avg_government_satisfaction', 'Avg_monthly_expense_when_under_age_21',
                                             'Avg_monthly_household_cost', 'Avg_monthly_income_all_years',
                                             'Last_school_grades', 'Overall_happiness_score', 'Political_interest_Total_Score',
                                             'Weighted_education_rank', 'Yearly_ExpensesK', 'Yearly_IncomeK']
        scalar_0_to_1 = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scalar_minus1_to_1 = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        for feature in features_to_normalize_0_to_1:
            df[feature] = scalar_0_to_1.fit_transform(df[feature].values.reshape(-1, 1))

        for feature in features_to_normalize_minus1_to_1:
            df[feature] = scalar_minus1_to_1.fit_transform(df[feature].values.reshape(-1, 1))

    return df


def test_with_random_forest(df_to_train_on, df_to_test_on, labels_are_strings=True):
    '''
        Evaluate with Random Forest Classifier.
        Bear in mind that the "df" parameter should be either the VALIDATION or TEST set.
        It does not have a default value to emphasize that fact.
    '''

    if labels_are_strings:
        score_function = metrics.accuracy_score
    else:
        score_function = metrics.mean_squared_error
        warnings.warn('If labels are ints then the problem is tested like regression and not classification!!')

    df_train_features, df_train_label = drop_label_column(df_to_train_on), get_label_column(df_to_train_on)
    df_test_features, df_test_label = drop_label_column(df_to_test_on), get_label_column(df_to_test_on)

    random_forest = RandomForestClassifier(n_estimators=100, random_state=41, max_depth=3)
    random_forest.fit(df_train_features, df_train_label)

    train_predition = random_forest.predict(df_train_features)
    print('Accuracy on training set: {}'.format(score_function(df_train_label, train_predition)))

    test_pred = random_forest.predict(df_test_features)
    print('Accuracy on test set: {}'.format(score_function(df_test_label, test_pred)))

