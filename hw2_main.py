from globals import *
import wrapper_method_tests as wrapper_tests
import filtrer_method_tests as filter_tests
import utils


def clean_func(): return wrapper_tests.clean_data(*clean_args.values())

# df to use. could be train/validation/test
df = df_train
features_info_dict = utils.get_features_info_dict(df)

# Configs
to_clean = True
clean_args = {'df': df_train, 'features_info_dict': features_info_dict, 'drop_features': False, 'negative_to_mean': False, 'labels_to_unique_ints': False,
                    'nominal_to_bool_split': True, 'missing_values_fill': True, 'binary_to_numeric': True, 'normalization': False}

to_print_corr_matrix = False  # RUN WITHOUT CLEANING (except for making everything numeric)
correlation_thresholds, print_info = [0.8], False

to_test_accuracy = False
df_to_train, df_to_test, clean_accuracy_dfs = (df_train, df_validation, True)

to_test_feature_selection = True
feature_selection_method = filter_tests.sfs

to_plot_two_features_together = False
x_feature, y_feature = ('Yearly_ExpensesK', 'Avg_Residancy_Altitude')

to_plot_feature_and_label = False
label_to_plot = 'Purples'

# --------------------- RUN -----------------------------

if to_clean:
    # x = list(df.columns)
    df = clean_func()
    # y = list(df.columns)
    # print([elem for elem in x if elem not in y])
    # print(len(x)-len(y))

if to_print_corr_matrix:
    for threshold in correlation_thresholds:
        filter_tests.find_correlated_features(df=df, correlation_threshold=threshold)

if to_test_accuracy:
    clean_args['df'] = df_to_train
    cleaned_train = clean_func()
    clean_args['df'] = df_to_test
    cleaned_val = clean_func()
    clean_args['df'] = df

    missing_cols = [col for col in cleaned_train.columns if col not in cleaned_val]
    cleaned_train.drop(columns=missing_cols)

    wrapper_tests.test_with_random_forest(cleaned_train, cleaned_val)

if to_test_feature_selection:
    res = feature_selection_method(df=df)
    print('{} features sorted by importance:'.format(len(res)))
    for elem in res:
        print(elem)

if to_plot_two_features_together:
    import matplotlib.pyplot
    matplotlib.pyplot.plot(df[x_feature], df[y_feature], 'ro')
    utils.plt.show()

if to_plot_feature_and_label:
    utils.plot_label_and_examples(label_to_plot, df)
