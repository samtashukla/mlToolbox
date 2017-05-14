#!/Users/sgagnon/anaconda/bin/python

import numpy as np
import scipy as sp
import pandas as pd

print 'v1'

# Preproc stuff
###########################
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, Imputer

# Models
###########################
from sklearn import linear_model
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
                             GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# CV
###########################
from sklearn.model_selection import KFold, StratifiedKFold

# Evaluation
###########################
from sklearn.metrics import mean_squared_error, median_absolute_error, \
                            roc_auc_score, f1_score, precision_score, recall_score

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

###############################################
# Model fitting
###############################################
def fit_evaluate_models(X, y, dv_type, models, n_cv_folds=2, scale_x=False, n_poly=False):
    ''' Fit and evaluate models
    models: dict, example: models = {'ols': ols, 'ridge': ridge, 'grad_boost': gboost}

    '''

    # Set up dataframe to store output + CV scheme
    # Is the DV numeric or categorical?
    if dv_type == 'numeric':
        df_eval = pd.DataFrame(columns=['model', 'eval_type', 'r2', 'mse', 'med_abs_e'])

        kf = KFold(n_splits=n_cv_folds, shuffle=True)
        cv = kf.split(X)

    elif dv_type == 'categorical':
        df_eval = pd.DataFrame(columns=['model', 'eval_type', 'acc', 'auc', 'f1'])

        skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True)
        cv = skf.split(X, y)

    # Fit to training, score on test data
    for i, (train, test) in enumerate(cv):
        print 'CV fold ' + str(i + 1) + ' ...'

        # Segment into training/testing using cv scheme
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        y_train = y[train]
        y_test = y[test]

        # Optionally add in interactions/n order polynomial features
        if n_poly:
            poly_feat = PolynomialFeatures(degree=n_poly, include_bias=False)
            X_train = poly_feat.fit_transform(X_train)
            X_test = poly_feat.fit_transform(X_test)

        # Optionally scale features robustly
        if scale_x:
            robust_scaler = RobustScaler()
            X_train = robust_scaler.fit_transform(X_train)
            X_test = robust_scaler.transform(X_test)

        # Evaluate
        for model_name, model in models.items():

            # Fit the model
            model.fit(X_train, y_train)

            # Evaluate, iterating through training/testing
            for xs, ys, eval_type in zip([X_train, X_test],
                                         [y_train, y_test],
                                         ['train', 'test']):

                if dv_type == 'numeric':

                    print 'Mean prediction ('+eval_type+ ') : '
                    print np.mean(model.predict(xs))

                    # med_abs_e: robust to outliers
                    row = {'model': model_name,
                           'eval_type': eval_type,
                           'r2': model.score(xs, ys),
                           'mse': mean_squared_error(ys, model.predict(xs)),
                           'med_abs_e': median_absolute_error(ys, model.predict(xs))}
                elif dv_type == 'categorical':

                    # Binary classification
                    if len(y.unique()) == 2:
                        if model_name not in ['random forest', 'knn (5)']:
                            y_score = model.decision_function(xs)

                            row = {'model': model_name,
                                   'eval_type': eval_type,
                                   'acc': model.score(xs, ys),
                                   'auc': roc_auc_score(ys, y_score),
                                   'f1': f1_score(ys, model.predict(xs)),
                                   'precision': precision_score(ys, model.predict(xs)),
                                   'recall': recall_score(ys, model.predict(xs))}
                        else:
                            row = {'model': model_name,
                                   'eval_type': eval_type,
                                   'acc': model.score(xs, ys),
                                   'auc': np.nan,
                                   'f1': f1_score(ys, model.predict(xs)),
                                   'precision': precision_score(ys, model.predict(xs)),
                                   'recall': recall_score(ys, model.predict(xs))}
                    # >2 class classification
                    else:
                        row = {'model': model_name,
                               'eval_type': eval_type,
                               'acc': model.score(xs, ys),
                               'f1': f1_score(ys, model.predict(xs), average='weighted'),
                               'precision': precision_score(ys, model.predict(xs),
                                                            average='weighted'),
                               'recall': recall_score(ys, model.predict(xs),
                                                      average='weighted')}

                # Append to dataframe
                df_eval = df_eval.append(row, ignore_index=True)

    return df_eval
