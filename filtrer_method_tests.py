from utils import *
from sklearn.feature_selection import VarianceThreshold


def find_quasi_constant_features(df=ElectionData, variance_threshold=0.1, to_print=True):
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


def find_duplicated_features(df=ElectionData, to_print=True):
    if to_print:
        print('\nStarting duplicated features test.')

    df_features = drop_label_column(df)
    df_features_T = df_features.T

    unique_features = df_features_T.drop_duplicates(keep='first').T
    duplicated_features = [dup_col for dup_col in df_features.columns if dup_col not in unique_features.columns]

    if to_print:
        print('Found {} duplicated features. The found features are:\n{}\n'.format(len(duplicated_features), duplicated_features))

    return duplicated_features


def find_correlated_features(df=ElectionData, correlation_threshold=0.8, to_print=True):
    if to_print:
        print('\nStarting correlated features test with threshold of {}.'.format(correlation_threshold))

    df_features = drop_label_column(df)
    features_names = list(df_features.columns)

    correlated_features = set()
    correlation_matrix = df_features.corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                if to_print:
                    print('Correlation found between {} and {}.'.format(features_names[i], features_names[j]))
                correlated_features.add(correlation_matrix.columns[i])

    if to_print:
        print('\nFound {} correlated_features. The found features (to drop) are:\n{}\n'.format(len(correlated_features), correlated_features))

    return correlated_features

