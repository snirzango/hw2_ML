import numpy as np
import pandas as pn


def split_nominal_feature_to_bool_features(df, feature_to_split):
    feature_values = df[feature_to_split].unique()

    for issue in feature_values:
        issue_bool_list = [1 if x == issue else 0 for x in df.Most_Important_Issue]
        df['{}_{}'.format(feature_to_split, issue)] = issue_bool_list


def get_features_probabilities_dict(df, eliminate_nd_elements = True):
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
            for value in feature_info_dict.values():
                if isinstance(value, np.ndarray):
                    value = value.tolist()

        features_dict[feature] = feature_info_dict

    return features_dict


def fill_missing_values(prob_Dict, feature_name):
    names = prob_Dict[feature_name]['names']
    probs = prob_Dict[feature_name]['probs']

    return np.random.choice(names, 1, p=probs)

