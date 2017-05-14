#!/Users/sgagnon/anaconda/bin/python

import numpy as np
import scipy as sp
import pandas as pd

###############################################
# Functions to create polynomial variables
###############################################
def poly(x, degree=2):
    """Return polynomials up to nth degree
    Perform QR decomp of a matrix up to the degree-th power
    Returns the Q matrix, -1st column (like R's poly())
    http://stackoverflow.com/questions/41317127/python-equivalent-to-r-poly-function
    """
    x = np.array(x)
    X_trans = np.transpose(np.vstack((x**k for k in range(degree + 1))))
    return np.linalg.qr(X_trans)[0][:, 1:]

def add_poly_features(data, columns, degree=2):
    """Add polynomial features to df
    note: only works for degree == 2 right now

    Args:
    data: pandas dataframe
    columns: list of column names (as strings)
    degree: int specifying degree up to which you want added
    
    Returns:
    pandas dataframe with new column for polynomial trend
    """

    if degree != 2:
        print 'Only works w/2 degrees right now...'
        return

    for col in columns:
        new_col = col + '_poly' + str(degree)
        data[new_col] = np.nan
        data[[col, new_col]] = poly(data[col], degree=degree)

    return data

###############################################
# Functions to clean data
###############################################
def winsorize_features(data, columns, limit=0.025):
    """Winsorize the columns to account for outliers

    Args:
    data: pandas dataframe
    columns: list of column names (as strings)
    limit: float (0 to 1) speciying the percentage to cut on each side of the column
    """
    for col in columns:
        data[col] = sp.stats.mstats.winsorize(data[col], limits=[limit, limit])

def log_features(data, columns):
    """Log transform the columns

    Args:
    data: pandas dataframe
    columns: list of column names (as strings)
    """
    for col in columns:
        # deal with 0/1 values
        if np.sum(data[col] == 0) > 0:
            print 'Replacing 0s with 0.025...'
            data.loc[data[col] == 0, col] = 0.025

        data[col] = np.log(data[col])


def logit_features(data, columns, upper_bound=1):
    """Logit transform the columns

    Args:
    data: pandas dataframe
    columns: list of column names (as strings)
    upper_bound: int, usually 1 but could be something like 100
    """
    for col in columns:

        if upper_bound != 1:
            print 'Rescaling data...'
            data[col] = data[col] / upper_bound

        # deal with 0/1 values
        if np.sum(data[col].isin([0, 1])) > 0:
            print 'Replacing 0s with 0.025, 1s with 0.925...'
            data.loc[data[col] == 0, col] = 0.025
            data.loc[data[col] == 1, col] = 0.925

        data[col] = sp.special.logit(data[col])

###############################################
# Functions to impute missing data
###############################################
def fillna_mode(data, columns, verbose=True):
    """Fill nas with the mode of the column

    Args:
    data: pandas dataframe
    columns: list of column names (as strings)
    verbose: bool; if True, print out what value you're using for replacement
    """
    for col in columns:
        fill_val = data[col].mode()[0]
        if verbose: print 'Filling ' + col + ' with: ' + fill_val
        data[col].fillna(fill_val, inplace=True)

def fillna_median(data, columns, grouping=False, val='median', verbose=True):
    """Fill nas with the mode of the column

    Args:
    data: pandas dataframe
    columns: list of column names (as strings)
    grouping: list of column names on which to perform grouping
    val: str specifying if "median" or other func acceptable for df.transform()
    verbose: bool; if True, print out what value(s) you're using for replacement
    """
    for col in columns:
        if grouping:
            data[col].fillna(data.groupby(grouping)[col].transform(val), inplace=True)
            meds = data.groupby(grouping)[col].median()
        else:
            meds = data[col].median()
            data[col].fillna(meds, inplace=True)
        if verbose:
            print 'Medians: '
            print meds
