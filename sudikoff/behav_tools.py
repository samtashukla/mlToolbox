"""Functions for behavioral analyses

The functions in this module help run stats for behav analyses

"""

from __future__ import division
from scipy.stats import norm
from math import exp,sqrt
Z = norm.ppf
import pandas as pd

def calc_sdt(data, coding_dict=None, measures=None):

    """Calculate signal detection stats for a pandas df
    Parameters
    ----------
    data : Pandas dataframe
        dataframe including cols for subject, objective status of each trial (e.g., old, new),
        response for each trial (e.g., old, new)
    coding_dict : dict
        dictionary with information about objective (objective_col; string) and
        response columns (response_col; string), subject column (subj_col; string),
        objective old (old; list of strings) and new (new; list of strings) labels,
        and subjective "old" response labels (old_resp; list of strings).
            Example:
            coding_dict = dict(objective_col='TrialType', # column name for new
                       old=['old'], # objectively old label
                       new=['similar', 'new'], # objectively new label
                       response_col='Resp_bin', #
                       old_resp=['Old'],
                       subj_col='Subject',
                       )
    measures : list of strings
        list of SDT measures to include in output; options are 'd' (dprime), 'beta', 'c', and 'Ad'
    Returns
    -------
    df : Pandas dataframe
        A longform dataframe with a column for subject ID, measure, and value.
    """

    # get relevant info
    subj_col = coding_dict['subj_col']
    obj_col = coding_dict['objective_col']
    resp_col = coding_dict['response_col']
    old = coding_dict['old']
    new = coding_dict['new']
    old_resp = coding_dict['old_resp']

    # init new dataframe
    df = pd.DataFrame(columns=[subj_col, 'measure', 'value'])

    # calculate dprime for each subj
    for subj in data[subj_col].unique():

        data_s = data[data[subj_col] == subj]
        count_old = data_s[data_s[obj_col].isin(old)].Trial.count()
        count_new = data_s[data_s[obj_col].isin(new)].Trial.count()
        count_hit = data_s[data_s[obj_col].isin(old) &
                           data_s[resp_col].isin(old_resp)].Trial.count()
        count_fa = data_s[data_s[obj_col].isin(new) &
                          data_s[resp_col].isin(old_resp)].Trial.count()

        # Floors an ceilings are replaced by half hits and half FA's
        halfHit = 0.5/count_old
        halfFa = 0.5/count_new

        # Calculate hitrate and avoid d' infinity
        hitRate = count_hit/count_old
        if hitRate == 1: hitRate = 1-halfHit
        if hitRate == 0: hitRate = halfHit

        # Calculate false alarm rate and avoid d' infinity
        faRate = count_fa/count_new
        if faRate == 1: faRate = 1-halfFa
        if faRate == 0: faRate = halfFa

        out = {}
        out['d'] = Z(hitRate) - Z(faRate)
        out['beta'] = exp(Z(faRate)**2 - Z(hitRate)**2)/2
        out['c'] = -(Z(hitRate) + Z(faRate))/2
        out['Ad'] = norm.cdf(out['d']/sqrt(2))

        for measure in measures:
            row = pd.Series({subj_col: subj,
                             'measure': measure,
                             'value': out[measure]})
            df = df.append(row, ignore_index=True)

    return df
