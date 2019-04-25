from utils import *

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector


def feature_selection_rfe_test(df=df_train, RFE_test=True, nonlinear_test=True):
    df_features = drop_label_column(df)
    df_label = df[label_name]
    features_names = list(df_features.columns)

    # Configurations:
    num_of_all_features = len(df_features.columns)
    max_features_to_select = int((num_of_all_features / 2) + 4)
    min_features_to_select = 1

    # Alpha (regularization strength) of LASSO regression
    lasso_eps = 0.001
    lasso_nalpha = 200
    lasso_iter = 9000

    # Min and max degree of polynomials features to consider
    degree_min = 2
    degree_max = 8

    results = {test_mode: {'optimal_features_number': 0, 'best_score': 0, 'features_selected': [], 'features_dropped': []}
               for test_mode in ['linear', 'nonlinear']}

    print('Starting feature selection test.')
    print('Total features: {}, max features to select: {}, min features to select: {}.\n'.format(
                                                                num_of_all_features, max_features_to_select, min_features_to_select))

    x_train, x_test, y_train, y_test = train_test_split(df_features, df_label, test_size=0.3, random_state=0)

    if RFE_test:
        print('Starting linear test.')
        for i in range(1, max_features_to_select):
            model = LinearRegression()
            rfe = RFE(model, n_features_to_select=i)
            x_train_rfe = rfe.fit_transform(x_train, y_train)
            x_test_rfe = rfe.transform(x_test)
            model.fit(x_train_rfe, y_train)
            score = model.score(x_test_rfe, y_test)

            print('Testing selection of {} features. Score: {}.'.format(i, score))

            if score > results['linear']['best_score']:
                results['linear']['best_score'], results['linear']['optimal_features_number'] = score, i
                results['linear']['features_selected'] = [features_names[i] for i in range(len(features_names)) if rfe.support_[i]]
                results['linear']['features_dropped'] = [features_names[i] for i in range(len(features_names)) if not rfe.support_[i]]

        print('\nLinear test ended. Summary:')
        print('Best score: {}, number of features: {}.\n'.format(results['linear']['best_score'], results['linear']['optimal_features_number']))
        print('Features selected: {}.\n'.format(results['linear']['features_selected']))
        print('Features dropped: {}.\n'.format(results['linear']['features_dropped']))

    # if nonlinear_test:
    #     # Make a pipeline model with polynomial transformation and LASSO regression with cross-validation,
    #     # run it for increasing degree of polynomial (complexity of the model)
    #     for degree in range(degree_min, degree_max + 1):
    #         model = make_pipeline(preprocessing.PolynomialFeatures(degree, interaction_only=False),
    #                               LassoCV(cv=5, tol=0.001, max_iter=2000))  # TODO: check 2000
    #         model.fit(x_train, y_train)
    #         test_pred = np.array(model.predict(x_test))
    #         RMSE = np.sqrt(np.sum(np.square(test_pred - y_test)))
    #         soore = model.score(x_test, y_test)

    print('\nAll Tests Ended.')


def step_forward_selection_by_random_forest(features_to_select=27, df=df_train, to_print=True):

    if to_print:
        print('\nStarting step forward feature selection test using RandomForest classifier.')

    df_features = drop_label_column(df)
    df_label = get_label_column(df)

    feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1, n_estimators=100),
                                                 k_features=features_to_select, forward=True, verbose=2, cv=4)
    features = feature_selector.fit(df_features, df_label)
    filtered_features = df_features.columns[list(features.k_feature_idx_)]

    if to_print:
        print('Found {} features to drop. Features are: \n{}'.format(len(filtered_features), filtered_features))

    return filtered_features

    # This algorithm is very heavy and takes a lot of time to run. Its last results were:
    #
    # Index(['Most_Important_Issue_Education', 'Most_Important_Issue_Social',
    #        'Most_Important_Issue_Environment', 'Most_Important_Issue_Financial',
    #        'Most_Important_Issue_Military', 'Most_Important_Issue_Foreign_Affairs',
    #        'Will_vote_only_large_party_No', 'Main_transportation_nan',
    #        'Occupation_Hightech', 'Occupation_Industry_or_other',
    #        'Occupation_Services_or_Retail', 'Occupation_nan',
    #        'Avg_monthly_expense_when_under_age_21', 'AVG_lottary_expanses',
    #        'Avg_environmental_importance', 'Married', 'Avg_Residancy_Altitude',
    #        'Avg_government_satisfaction', 'Avg_monthly_household_cost',
    #        'Phone_minutes_10_years', 'Avg_size_per_room',
    #        'Weighted_education_rank', 'Last_school_grades',
    #        'Political_interest_Total_Score', 'Number_of_valued_Kneset_members',
    #        'Avg_education_importance', 'Financial_agenda_matters'],
    #         dtype='object')

