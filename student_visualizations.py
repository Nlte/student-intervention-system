import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt




def filter_data(data, condition):
    """
    Remove elements that do not match the condition provided.
    Takes a data list as input and returns a filtered list.
    Conditions should be a list of strings of the following format:
      '<field> <op> <value>'
    where the following operations are valid: >, <, >=, <=, ==, !=
    """

    field, op, value = condition.split(" ")

    # convert value into number or strip excess quotes if string
    try:
        value = float(value)
    except:
        value = value.strip("\'\"")

    # get booleans for filtering
    if op == ">":
        matches = data[field] > value
    elif op == "<":
        matches = data[field] < value
    elif op == ">=":
        matches = data[field] >= value
    elif op == "<=":
        matches = data[field] <= value
    elif op == "==":
        matches = data[field] == value
    elif op == "!=":
        matches = data[field] != value
    else:  # catch invalid operation codes
        raise Exception("Invalid comparison operator. Only >, <, >=, <=, ==, != allowed.")

    # filter data and outcomes
    data = data[matches].reset_index(drop=True)
    return data


def passed_stats(data, outcomes, key, filters=[]):
    """
    Print out selected statistics regarding survival, given a feature of
    interest and any number of filters (including no filters)
    """

    # Check that the key exists
    if key not in data.columns.values:
        print("'{}' is not a feature of the student data. Did you spell something wrong?".format(key))
        return False

    # Merge data and outcomes into single dataframe
    all_data = pd.concat([data, outcomes], axis=1)

    # Apply filters to data
    for condition in filters:
        all_data = filter_data(all_data, condition)

    # Create outcomes DataFrame
    all_data = all_data[[key, 'passed']]

    # Create plotting figure
    plt.figure(figsize=(8, 6))

    # 'Numerical' features
    # Divide the range of data into bins and count survival rates
    min_value = all_data[key].min()
    max_value = all_data[key].max()
    value_range = max_value - min_value

    bins = np.arange(0, all_data[].max(), 20)

    # Overlay each bin's survival rates
    nonpass_vals = all_data[all_data['Survived'] == 0][key].reset_index(drop = True)
    pass_vals = all_data[all_data['Survived'] == 1][key].reset_index(drop = True)
    plt.hist(nonpass_vals, bins=bins, alpha=0.6,
             color='red', label='Did not survive')
    plt.hist(pass_vals, bins=bins, alpha=0.6,
             color='green', label='Survived')
    # Add legend to plot
    plt.xlim(0, bins.max())
    plt.legend(framealpha=0.8)

    # Common attributes for plot formatting
    plt.xlabel(key)
    plt.ylabel('Number of Passengers')
    plt.title('Passenger Survival Statistics With \'%s\' Feature' % (key))
    plt.show()

    # Report number of passengers with missing values
    if sum(pd.isnull(all_data[key])):
        nan_outcomes = all_data[pd.isnull(all_data[key])]['Survived']
        print( "Passengers with missing '{}' values: {} ({} survived, {} did not survive)".format( \
              key, len(nan_outcomes), sum(nan_outcomes == 1), sum(nan_outcomes == 0)))
