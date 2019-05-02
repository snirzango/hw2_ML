from utils import *
from globals import *
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.metrics import mean_squared_error, zero_one_loss


def find_quasi_constant_features(df=df_train, variance_threshold=0.1, to_print=True):
    '''
        Find features that their values are 99% similar and therefore
        Are probably not relevant.

        Do this test WITHOUT normalizing the data!!
    '''

    if to_print:
        print('\nStarting Quansi-Constant features test by variance threshold of {}.'.format(variance_threshold))

    df_features = drop_label_column(df)
    qconstant_filter = VarianceThreshold(threshold=variance_threshold)
    qconstant_filter.fit(df_features)

    # Results:
    features_not_qconstants = list(df_features.columns[qconstant_filter.get_support(indices=True)])
    features_that_are_qconstant = [feature for feature in df_features.columns
                                   if feature not in features_not_qconstants]

    if to_print:
        print('Found {} quansi-constant features out of {} features total.'.format(len(features_that_are_qconstant), len(df_features.columns)))
        print('The found features are:\n{}\n'.format(features_that_are_qconstant))

    return features_that_are_qconstant


def find_duplicated_features(df=df_train, to_print=True):
    if to_print:
        print('\nStarting duplicated features test.')

    df_features = drop_label_column(df)
    df_features_T = df_features.T

    unique_features = df_features_T.drop_duplicates(keep='first').T
    duplicated_features = [dup_col for dup_col in df_features.columns if dup_col not in unique_features.columns]

    if to_print:
        print('Found {} duplicated features. The found features are:\n{}\n'.format(len(duplicated_features), duplicated_features))

    return duplicated_features


def find_correlated_features(df=df_train, correlation_threshold=0.8, to_print=True):
    if to_print:
        print('\n', '~'*10)
        print('Starting correlated features test with threshold of {}.\n'.format(correlation_threshold))

    df_features = drop_label_column(df)
    features_names = list(df_features.columns)

    correlated_features_info = []
    correlation_matrix = df_features.corr()

    found_features_count = 0

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                features_str = (features_names[i], features_names[j])

                # Preparing two numpy arrays with no nan values
                features_np = (df_features[features_str[0]].values, df_features[features_str[1]].values)  # Features as numpy arrays

                feature0_not_numeric, feature1_not_numeric = (features_np[0].dtype == np.object, features_np[1].dtype == np.object)
                if feature0_not_numeric or feature1_not_numeric:
                    # print('\n', '*'*5)
                    # print("find_correlated_features: [WARNING]! Skipping the following features since one or more aren't numeric:")
                    # print('{} (not numeric? {}), {} (not numeric? {})'.format(features_str[0], feature0_not_numeric,
                    #                                                           features_str[1], feature1_not_numeric))
                    # print('*' * 5, '\n')
                    continue

                mask = ~np.isnan(features_np[0]) & ~np.isnan(features_np[1])
                features_np = (features_np[0][mask], features_np[1][mask])  # Now containing no nan values

                slope, intercept, correlation_coefficient, _, _ = stats.linregress(features_np[0], features_np[1])

                MSE = mean_squared_error((features_np[0] * slope) + intercept, features_np[1])
                MSE = MSE / np.mean(features_np[1])  # normalize by division by max element

                info_dict = {'features': features_str, 'slope': slope, 'intercept': intercept, 'corr': correlation_coefficient, 'MSE': MSE}

                found_features_count += 1
                correlated_features_info.append(info_dict)

    correlated_features_info = sorted(correlated_features_info, key=lambda elem: elem['MSE'])

    if to_print:
        print('Found {} features:'.format(found_features_count))
        for dictionary in correlated_features_info:
            print(dictionary)

        print('\n*DONE*')
        print('~' * 10, '\n')

    return correlated_features_info


def select_k_best_features(df=df_train, k=30):
    df_features, df_label = (drop_label_column(df), get_label_column(df))
    k_best_clf = SelectKBest(k=k).fit(df_features, df_label)

    all_of_features = list(df_features.columns)

    features_score = k_best_clf.scores_
    features_mask = k_best_clf.get_support()
    dropped_features = []
    kept_features = []

    for feature, is_dropped, score in zip(all_of_features, features_mask, features_score):
        if is_dropped:
            dropped_features.append((feature, score))
        else:
            kept_features.append((feature, score))

    return sorted(kept_features, key=lambda tup: tup[1], reverse=True), sorted(dropped_features, key=lambda tup: tup[1], reverse=True)


def relief(df=df_train, iterations=3000, threshold=12345):

    w_vector = {feature: 0 for feature in [elem for elem in list(df.columns) if elem not in [label_name]]}

    for feature in w_vector.keys():
        print(feature)

        df_feature_and_label = df.drop(columns=[elem for elem in w_vector.keys() if elem not in [label_name, feature]])

        is_binary_feature = len(list(set(df[feature]))) == 2

        if is_binary_feature:
            number_of_samples = 70
            number_of_neighbours = 200

            miss, match = (0, 0)
            c = 0
            for _ in range(number_of_samples):

                if c in [20, 45, 60]:
                    print(c)
                c += 1

                sample = df_feature_and_label.sample()
                sample_label, sample_feature_value = sample[label_name], float(sample[feature])

                counter = 0

                while counter < number_of_neighbours:
                    neighbour = df_feature_and_label.sample()
                    neighbour_label, neighbour_feature_value = neighbour[label_name], float(neighbour[feature])

                    if neighbour_label.values[0] != sample_label.values[0]:
                        continue

                    counter += 1

                    if sample_feature_value == neighbour_feature_value:
                        match += 1
                    else:
                        miss += 1

            w_vector[feature] = match / (miss + match)

        else:

            for _ in range(iterations):

                random_row = df_feature_and_label.sample()
                sample_label, value = random_row[label_name].values[0], random_row[feature].values[0]

                values_of_same_vote = df_feature_and_label[df_feature_and_label[label_name] == sample_label][feature].values
                values_of_other_vote = df_feature_and_label[df_feature_and_label[label_name] != sample_label][feature].values

                idx_of_nearest_hit = (np.abs(values_of_same_vote - value)).argmin()
                idx_of_nearest_miss = (np.abs(values_of_other_vote - value)).argmin()

                nearest_hit = values_of_same_vote[idx_of_nearest_hit]
                nearest_miss = values_of_other_vote[idx_of_nearest_miss]

                w_vector[feature] += float(((value - nearest_miss) ** 2) - ((value - nearest_hit) ** 2))

    sorted_w_vector = [(k, w_vector[k]) for k in sorted(w_vector, key=w_vector.get, reverse=True)]
    return sorted_w_vector


