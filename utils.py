import numpy as np
import pandas as pn


def split_nominal_feature_to_bool_features(df, feature_to_split):
    feature_values = df[feature_to_split].unique()

    for issue in feature_values:
        issue_bool_list = [1 if x == issue else 0 for x in df.Most_Important_Issue]
        df['{}_{}'.format(feature_to_split, issue)] = issue_bool_list


def get_features_probabilities_dict(df):
    features = list(df.columns)  # features[0] is labels
    features_dict = {}

    # Counting and will later normalize by total_feature_size
    for feature in features:
        feature_prob_dict = {'value': np.array([]), 'probs': np.array([])}  # value/prob
        feature_values = df[feature].unique()
        feature_total_not_missing_values = 0

        for feature_value in feature_values:
            try:
                fitting_values_count = (df[feature] == feature_value).value_counts()[True]
            except KeyError:
                fitting_values_count = 0

            feature_total_not_missing_values += fitting_values_count
            np.append(feature_prob_dict['value'], feature_value)
            np.append(feature_prob_dict['probs'], fitting_values_count)
            # feature_prob_dict['value'].append(feature_value)
            # feature_prob_dict['probs'].append(fitting_values_count)

        # Normalize
        feature_prob_dict['probs'] /= feature_total_not_missing_values
        # feature_prob_dict = {k: v / feature_total_not_missing_values for k, v in feature_prob_dict.items()}
        features_dict[feature] = feature_prob_dict

    return features_dict


def fill_missing_values(prob_Dict, feature_name):
    names = prob_Dict[feature_name]['names']
    probs = prob_Dict[feature_name]['probs']

    return np.random.choice(names, 1, p=probs)

