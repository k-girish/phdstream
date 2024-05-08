import pandas as pd


def column_names_for_a_range(end, start=0):
    return ['col{}'.format(i) for i in range(start, end)]


def one_col_query(col, val, string_val=False, for_not_equal=False):
    if string_val:
        if for_not_equal:
            template = "({} != '{}')"
        else:
            template = "({} == '{}')"
    else:
        if for_not_equal:
            template = "({} != {})"
        else:
            template = "({} == {})"

    return template.format(col, val)


def create_query_for_given_cols_and_values(nullable_cols, nullable_vals, complement_las_col=False):
    cols = []
    vals = []

    # Removing nans
    for i in range(len(nullable_vals)):
        if (type(nullable_vals[i]) == list) or (not pd.isna(nullable_vals[i])):
            cols.append(nullable_cols[i])
            vals.append(nullable_vals[i])

    n_cols = len(cols)

    # Assumes that there is atleast one element in the cols and vals value
    ans = one_col_query(cols[0], vals[0], string_val=(type(vals[0]) is str),
                        for_not_equal=(complement_las_col and n_cols == 1))

    # Assumes that they both have the same length
    for i in range(1, n_cols):
        if type(vals[i]) is list:
            # Create 'or' condition on all the values
            # Assumes that this is not for a not equal task
            col_multi_val_query = one_col_query(cols[i], vals[i][0], string_val=(type(vals[i][0]) is str),
                                                for_not_equal=False)
            for j in range(1, len(vals[i])):
                curr_query = one_col_query(cols[i], vals[i][j], string_val=(type(vals[i][j]) is str),
                                           for_not_equal=False)
                col_multi_val_query = "{} or {}".format(col_multi_val_query, curr_query)

            # Create and condition with the overall query
            ans = "{} and ({})".format(ans, col_multi_val_query)
        else:
            curr_query = one_col_query(cols[i], vals[i], string_val=(type(vals[i]) is str),
                                       for_not_equal=(complement_las_col and (i == n_cols - 1)))
            ans = "{} and {}".format(ans, curr_query)
    return ans


# This method will prefix the columns with the string 'col'

def create_query_with_prefix_col_for_given_col_indices_and_values(cols, vals):
    # Assumes that there is atleast one element in the cols and vals value
    ans = '(col{} == {})'.format(cols[0], vals[0])

    # Assumes that they both have the same length
    for i in range(1, len(cols)):
        ans = '{} and (col{} == {})'.format(ans, cols[i], vals[i])
    return ans
