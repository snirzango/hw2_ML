from utils import *
from filtrer_method_tests import *
from wrapper_method_tests import *
from globals import *
import globals

cleaned_train = clean_data(df=df_train, negative_to_mean=False, labels_to_unique_ints=False, nominal_to_bool_split=True,
                           missing_values_fill=True, binary_to_nominal=True, normalization=False)

cleaned_val = clean_data(df=df_validation, negative_to_mean=False, labels_to_unique_ints=False, nominal_to_bool_split=True,
                         missing_values_fill=True, binary_to_nominal=True, normalization=False)

test_with_random_forest(cleaned_train, cleaned_val)
