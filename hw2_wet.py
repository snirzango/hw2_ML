from utils import *
from filtrer_method_tests import *

df = ElectionData

# train, test = train_test_split(df, test_size=0.1)
# train, validation = train_test_split(train, test_size=0.25)

# cleaned_df = utils.clean_data(negative_to_mean=False, labels_to_unique_ints=True, nominal_to_bool_split=True,
#                               missing_values_fill=True, binary_to_nominal=True, normalization=False)

find_correlated_features()
