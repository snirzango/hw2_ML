from utils import *
from globals import *
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from sklearn.metrics import mean_squared_error

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
    features_not_qconstants = list(df_features.columns[qconstant_filter.get_support()])
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

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                features_str = (features_names[i], features_names[j])

                # Preparing two numpy arrays with no nan values
                features_np = (df_features[features_str[0]].values, df_features[features_str[1]].values)  # Features as numpy arrays

                feature0_not_numeric, feature1_not_numeric = (features_np[0].dtype == np.object, features_np[1].dtype == np.object)
                if feature0_not_numeric or feature1_not_numeric:
                    print('\n', '*'*5)
                    print("find_correlated_features: [WARNING]! Skipping the following features since one or more aren't numeric:")
                    print('{} (not numeric? {}), {} (not numeric? {})'.format(features_str[0], feature0_not_numeric,
                                                                              features_str[1], feature1_not_numeric))
                    print('*' * 5, '\n')
                    continue

                mask = ~np.isnan(features_np[0]) & ~np.isnan(features_np[1])
                features_np = (features_np[0][mask], features_np[1][mask])  # Now containing no nan values

                slope, intercept, correlation_coefficient, _, _ = stats.linregress(features_np[0], features_np[1])

                MSE = mean_squared_error((features_np[0] * slope) + intercept, features_np[1])
                MSE = MSE / np.amax(features_np[1])  # normalize by division by max element

                info_dict = {'features': features_str, 'slope': slope, 'intercept': intercept, 'corr': correlation_coefficient, 'MSE': MSE}

                correlated_features_info.append(info_dict)

    correlated_features_info = sorted(correlated_features_info, key=lambda elem: elem['MSE'])

    if to_print:
        print('Found features:')
        for dictionary in correlated_features_info:
            print(dictionary)

        print('\n*DONE*')
        print('~' * 10, '\n')

    return correlated_features_info
