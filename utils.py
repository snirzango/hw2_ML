import numpy as np
import pandas as pn

import matplotlib as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn import preprocessing
from sklearn import metrics


def split_nominal_feature_to_bool_features(df, feature_to_split):
    feature_values = df[feature_to_split].unique()

    for issue in feature_values:
        issue_bool_list = [1 if x == issue else 0 for x in df.Most_Important_Issue]
        df['{}_{}'.format(feature_to_split, issue)] = issue_bool_list

    df.drop(feature_to_split, axis=1)


def fill_missing_values(df, features_info_dict):
    features = list(df.columns)
    for feature in features:
        df[feature] = df[feature].apply(lambda v: choose_mean_value(features_info_dict, feature) if str(v) == 'nan' else v)


def get_features_probabilities_dict(df, eliminate_nd_elements=True):
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
        return np.random.choice(names, 1, p=probs)
    else:
        return features_info_dict[feature_name]['mean']


def test(x_train, y_train, x_test, y_test, classifier=RandomForestClassifier(n_estimators=3)):

    classifier = classifier.fit(x_train, y_train)

    # output = forest.predict(test_data_noNaN)
    y_pred = classifier.predict(x_test)

    print('accuracy:', metrics.accuracy_score(y_test, y_pred))
    print('precision:', metrics.precision_score(y_test, y_pred))
    print('recall:', metrics.recall_score(y_test, y_pred))
    print('f1 score:', metrics.f1_score(y_test, y_pred))


def replace_negatives_with_mean(df, features):
    for feature in features:
        mean = df[feature][df[feature] >= 0].mean()
        df[feature] = df[feature].apply(lambda v: mean if v < 0 else v)


def feature_hist(df, feature):
    df.hist(column=feature, bins=100)
    plt.pyplot.show()


def find_outliers(df):
    ObjFeat = df.keys()[df.dtypes.map(lambda x: x == 'object')]
    for f in ObjFeat:
        df[f] = df[f].astype("str")
        df[f] = df[f].astype("category")
        df[f] = df[f].cat.rename_categories(range(df[f].nunique())).astype(int)
        df.loc[df[f].isnull(), f] = np.nan  # fix NaN conversion
    X_train = df.drop('Vote', axis=1)
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


def clean_data(df):
    ''' Main function to clean data '''

    #  Replacing negative numbers with mean for appropriate columns:
    features_to_apply = ['Avg_monthly_expense_when_under_age_21']
    replace_negatives_with_mean(df, features_to_apply)

    #  Nominal features splitting:
    nominal_features_to_split = []
    for feature in nominal_features_to_split:
        split_nominal_feature_to_bool_features(df, feature)

    features_info_dict = get_features_probabilities_dict(df)

    #  Fill missing values by probability/mean:
    fill_missing_values(df, features_info_dict)

    #  Normalization phase:
    features_to_normalize_0_to_1 = []
    features_to_normalize_minus1_to_1 = []
    scalar_0_to_1 = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scalar_minus1_to_1 = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    for feature in features_to_normalize_0_to_1:
        df[feature] = scalar_0_to_1.fit_transform(df[feature].values.reshape(-1, 1))

    for feature in features_to_normalize_minus1_to_1:
        df[feature] = scalar_minus1_to_1.fit_transform(df[feature].values.reshape(-1, 1))


