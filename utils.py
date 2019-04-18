import numpy as np
import pandas as pn
import category_encoders as ce
import matplotlib as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


def fill_missing_values(df, features_info_dict):
    features = list(df.columns)
    for feature in features:
        df[feature] = df[feature].apply(lambda v: choose_mean_value(features_info_dict, feature) if str(v) == 'nan' else v)
    return df


def get_features_info_dict(df, eliminate_nd_elements=True):
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


def feature_hist(df, feature, bins=100):
    df.hist(column=feature, bins=bins)
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


def feature_selection_RFE_test(df):
    df_features = df.drop('Vote', axis=1)
    df_label = df['Vote']

    num_of_all_features = len(df_features.columns)
    max_features_to_select = int((num_of_all_features / 2) + 4)
    min_features_to_select = 1

    optimal_features_number, best_score = 0, 0

    print('Starting feature selection test.')
    print('Total features: {}, max features to select: {}, min features to select: {}.\n'.format(
                                                                num_of_all_features, max_features_to_select, min_features_to_select))

    for i in range(1, max_features_to_select):
        x_train, x_test, y_train, y_test = train_test_split(df_features, df_label, test_size=0.3, random_state=0)
        model = LinearRegression()
        rfe = RFE(model, n_features_to_select=i)
        x_train_rfe = rfe.fit_transform(x_train, y_train)
        x_test_rfe = rfe.transform(x_test)
        model.fit(x_train_rfe, y_train)
        score = model.score(x_test_rfe, y_test)

        print('Testing selection of {} features. Score: {}.'.format(i, score))

        if score > best_score:
            optimal_features_number, best_score = i, score

    print('\nTest Ended.')
    print('Best choice found: {} features with score: {}.'.format(optimal_features_number, best_score))


def clean_data(df, features_info_dict=None):
    ''' Main function to clean data '''

    # TODO: add "features to drop" section.

    # Replacing negative numbers with mean for appropriate columns:
    features_to_apply = ['Avg_monthly_expense_when_under_age_21']
    replace_negatives_with_mean(df, features_to_apply)

    # Convert Vote (label) to unique numbers
    df['Vote'] = df['Vote'].astype("str")
    df['Vote'] = df['Vote'].astype("category")
    df['Vote'] = df['Vote'].cat.rename_categories(range(df['Vote'].nunique())).astype(int)
    df.loc[df['Vote'].isnull(), 'Vote'] = np.nan  # fix NaN conversion

    # Nominal features splitting:
    nominal_features_to_split = ['Most_Important_Issue', 'Will_vote_only_large_party', 'Main_transportation',
                                 'Occupation']
    df = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True, cols=nominal_features_to_split).fit_transform(df)

    if features_info_dict is None:
        features_info_dict = get_features_info_dict(df)

    #  Fill missing values by probability/mean:
    df = fill_missing_values(df, features_info_dict)

    # Binary nominal (yes/no etc') to numeric (0/1):
    binary_features_and_values = {'Looking_at_poles_results': {'Yes': 0, 'No': 1},
                                  'Financial_agenda_matters': {'Yes': 0, 'No': 1},
                                  'Married': {'Yes': 0, 'No': 1},
                                  'Gender': {'Female': 0, 'Male': 1},
                                  'Voting_Time': {'By_16:00': 0, 'After_16:00': 1},
                                  'Age_group': {'Below_30': 0, '30-45': 1, '45_and_up': 2}}
    for feature, replacement_dict in binary_features_and_values.items():
        df[feature] = df[feature].map(replacement_dict)

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
