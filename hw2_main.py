from wrapper_method_tests import *
from globals import *


def clean_func(): return clean_data(*clean_args.values())


# df to use. could be train/validation/test
df = df_train

# Configs
to_clean = True
clean_args = {'df': df, 'features_info_dict': None, 'negative_to_mean': False, 'labels_to_unique_ints': False,
              'nominal_to_bool_split': True, 'missing_values_fill': True, 'binary_to_nominal': True, 'normalization': False}

to_test_accuracy = True
df_to_train, df_to_test, clean_accuracy_dfs = (df_train, df_validation, True)

to_test_feature_selection = False
feature_selection_method = step_forward_selection_by_random_forest

# --------------------- RUN -----------------------------

if to_clean:
    df = clean_func()

if to_test_accuracy:
    clean_args['df'] = df_to_train
    cleaned_train = clean_func()
    clean_args['df'] = df_to_test
    cleaned_val = clean_func()
    clean_args['df'] = df

    test_with_random_forest(cleaned_train, cleaned_val)


if to_test_feature_selection:
    feature_selection_method(df=df)

