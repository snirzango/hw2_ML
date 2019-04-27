from utils import *
from globals import *
from sklearn.feature_selection import VarianceThreshold
from scipy import stats


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
                features_np = (df[features_str[0]].values, df[features_str[1]].values)  # Features as numpy arrays
                mask = ~np.isnan(features_np[0]) & ~np.isnan(features_np[1])
                features_np = (features_np[0][mask], features_np[1][mask])  # Now containing no nan values

                slope, intercept, correlation_coefficient, _, _ = stats.linregress(features_np[0], features_np[1])

                info_dict = {'features': features_str, 'slope': slope, 'intercept': intercept, 'corr': correlation_coefficient}

                correlated_features_info.append(info_dict)

                if to_print:
                    print('*{}* IS CORRELATED WITH *{}*.'.format(features_str[0], features_str[1]))
                    print('Slope: {}. Intercept: {}. Correlation Coefficient: {}.'.format(slope, intercept, correlation_coefficient))
                    print('~'*5)
    
    if to_print:
        print('*DONE*')
        print('~' * 10, '\n')

    return correlated_features_info
