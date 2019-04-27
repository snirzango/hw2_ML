from globals import *
import wrapper_method_tests as wrapper_tests
import filtrer_method_tests as filter_tests
import utils


def clean_func(): return wrapper_tests.clean_data(*clean_args.values())


# df to use. could be train/validation/test
df = df_train

# Configs
to_clean = True
clean_args = {'df': df, 'features_info_dict': None, 'negative_to_mean': False, 'labels_to_unique_ints': False,
              'nominal_to_bool_split': True, 'missing_values_fill': True, 'binary_to_nominal': True, 'normalization': False}

to_test_accuracy = False
df_to_train, df_to_test, clean_accuracy_dfs = (df_train, df_validation, True)

to_test_feature_selection = False
feature_selection_method = wrapper_tests.step_forward_selection_by_random_forest

to_print_corr_matrix = True
correlation_threshold, print_info = 0.8, False

# --------------------- RUN -----------------------------

if to_clean:
    df = clean_func()

if to_test_accuracy:
    clean_args['df'] = df_to_train
    cleaned_train = clean_func()
    clean_args['df'] = df_to_test
    cleaned_val = clean_func()
    clean_args['df'] = df

    wrapper_tests.test_with_random_forest(cleaned_train, cleaned_val)


if to_test_feature_selection:
    feature_selection_method(df=df)

if to_print_corr_matrix:
    correlated_features = filter_tests.find_correlated_features(df=df, correlation_threshold=correlation_threshold, to_print=print_info)
    print(correlated_features)

