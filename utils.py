from globals import *

import numpy as np
import pandas as pn
import matplotlib as plt

import warnings

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score

from matplotlib import pyplot as plt

def drop_label_column(df=df_train):
    return df.drop(label_name, axis=1)


def get_label_column(df=df_train):
    return df[label_name]


def fill_missing_values_by_feature_mean(features_info_dict, df=df_train):
    features = list(df.columns)
    for feature in features:
        df[feature] = df[feature].apply(lambda row: choose_mean_value(features_info_dict, feature) if str(row) == 'nan' else row)
    return df


def fill_missing_values_by_linear_connection(feature1, feature2, correlated_features_info, features_info_dict, df=df_train):
    if feature1 not in df.columns or feature2 not in df.columns:
        return df

    feature1_nulls_count = df[feature1].isnull().sum()
    feature2_nulls_count = df[feature2].isnull().sum()

    if feature1_nulls_count > feature2_nulls_count:
        feature_to_drop, feature_to_fill = feature1, feature2
    else:
        feature_to_drop, feature_to_fill = feature2, feature1

    for dictionary in correlated_features_info:
        if dictionary['features'] == (feature_to_drop, feature_to_fill):
            slope, intercept = (dictionary['slope'], dictionary['intercept'])
        elif dictionary['features'] == (feature_to_fill, feature_to_drop):
            slope, intercept = (dictionary['slope'] ** -1, dictionary['intercept']/dictionary['slope'])

    # df[feature_to_fill] = df[feature_to_fill].apply(
    #     lambda row: (df[feature_to_drop] * slope) + intercept if np.isnan(row) and not np.isnan(df[feature_to_drop])
    #     else choose_mean_value(features_info_dict, feature_to_drop) if np.isnan(row) and np.isnan(df[feature_to_drop])
    #     else row
    # )
    df[feature_to_fill].fillna((df[feature_to_drop] * slope) + intercept)

    df = df.drop(columns=[feature_to_drop])

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


def clean_data(df=df_train, features_info_dict=None, drop_features=True, negative_to_mean=True, labels_to_unique_ints=True,
               nominal_to_bool_split=True, missing_values_fill=True, binary_to_numeric=True, normalization=True):
    ''' Main function to clean data '''

    if drop_features:
        features_to_drop = ['%Of_Household_Income']
        df = df.drop(columns=features_to_drop)

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
        df = pn.get_dummies(df, columns=nominal_features_to_split)

    if features_info_dict is None:
        features_info_dict = get_features_info_dict(df)

    if missing_values_fill:
        #  Fill missing values by linear connection:
        import filtrer_method_tests

        MSE_threshold = 53

        correlated_features_info = filtrer_method_tests.find_correlated_features(df=df, to_print=False)
        to_fill_by_linear_connection = [dictionary['features'] for dictionary in correlated_features_info if dictionary['MSE'] <= MSE_threshold]

        for correlated_features in to_fill_by_linear_connection:
            df = fill_missing_values_by_linear_connection(correlated_features[0], correlated_features[1], correlated_features_info, features_info_dict, df)

        # Drop features with correlation that is not linear (remained ones):
        correlated_features_info = filtrer_method_tests.find_correlated_features(df=df, to_print=False)
        remained_correlated_features = set([dictionary['features'][0] for dictionary in correlated_features_info])
        df.drop(columns=remained_correlated_features)

        # Fill missing values by probability/mean:
        df = fill_missing_values_by_feature_mean(features_info_dict=features_info_dict, df=df)

    if binary_to_numeric:
        # Binary nominal (yes/no etc') to numeric (0/1):
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

        features_to_normalize_0_to_1 = [elem for elem in features_to_normalize_0_to_1 if elem in list(df.columns)]
        features_to_normalize_minus1_to_1 = [elem for elem in features_to_normalize_minus1_to_1 if elem in list(df.columns)]

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


def plot_label_and_examples(label, examples_list):
    examples_list = examples_list.values
    examples_indexes = []
    for example_index in range(examples_list.__len__()):
        if str(examples_list[example_index][24]) == label:
            examples_indexes.append(example_index)

    X = [examples_list[examples_indexes[i]] for i in range(examples_indexes.__len__())]
    for i in range(examples_indexes.__len__()):
        X[i] = np.delete(X[i], 24)
    T2 = [X[i].__len__() for i in range(examples_indexes.__len__())]
    T3 = [np.linspace(0, T2[i] - 1, T2[i]) for i in range(examples_indexes.__len__())]

    for i in range(len(X)):
        plt.plot(T3[i], X[i])

    plt.xlabel("manipulated features")
    plt.ylabel("value")
    plt.title("graph by " + label)
    plt.legend()
    plt.grid()
    plt.show()

