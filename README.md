# PyBehavToolbox
Python functions for manipulating pandas dataframes, useful for behavioral analyses and 
general data exploration.

## Dependencies (or just use Anaconda[(https://docs.continuum.io/)])
- numpy
- scipy
- pandas
- scikit-learn

## pandas_helpers.py
Contains various functions to preprocess pandas dataframes (e.g., transforming features, imputing missing data), as well as functions to fit and evaluate various models using `sklearn`. 

- `fit_evaluate_models`: Given samples (X) and a continuous or categorical vector to predict (y), train/test various models using a cross-validation approach, and return a pandas dataframe with evaluative metrics matching the type of dependent variable (e.g., coefficient of determination for a continuous DV, accuracy for a categorical DV). Optionally scale the features (across samples), or add in polynomial + interaction features.
