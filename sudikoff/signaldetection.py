"""Functions for signal detection theory

The functions in this module help calculate dprime and ROC curves

"""

from __future__ import division
from scipy.stats import norm
from math import exp,sqrt
Z = norm.ppf
import pandas as pd

def calc_sdt(data, coding_dict=None, measures=None):

    """Calculate signal detection stats (e.g., dprime, criterion, beta) from a pandas dataframe
    Parameters
    ----------
    data : Pandas dataframe
        longform dataframe including cols for subject, objective status of each trial (e.g., signal/old, noise/new),
        response for each trial (e.g., signal/old, noise/new)
    coding_dict : dict
        dictionary with information about objective column (objective_col; string) and
        response column (response_col; string), subject ID column (subj_col; string),
        objective "signal" (signal; list of strings) and "noise" (noise; list of strings) labels,
        and subjective "signal" response labels (signal_resp; list of strings).
            Example coding_dict (for a memory experiment):
            coding_dict = dict(objective_col='TrialType', # column name for new
                               signal=['old'], # objectively old label
                               noise=['similar', 'new'], # objectively new label
                               response_col='Resp_bin', #
                               signal_resp=['Old'],
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
    signal = coding_dict['old']
    noise = coding_dict['new']
    signal_resp = coding_dict['old_resp']

    # init new dataframe
    df = pd.DataFrame(columns=[subj_col, 'measure', 'value'])

    # calculate dprime for each subj
    for subj in data[subj_col].unique():

        data_s = data[data[subj_col] == subj]
        count_signal = data_s[data_s[obj_col].isin(signal)].Trial.count()
        count_noise = data_s[data_s[obj_col].isin(noise)].Trial.count()
        count_hit = data_s[data_s[obj_col].isin(signal) &
                           data_s[resp_col].isin(signal_resp)].Trial.count()
        count_fa = data_s[data_s[obj_col].isin(noise) &
                          data_s[resp_col].isin(signal_resp)].Trial.count()

        # Floors and ceilings are replaced by half hits and half FA's
        halfHit = 0.5/count_signal
        halfFa = 0.5/count_noise

        # Calculate hitrate, avoiding d' infinity
        hitRate = count_hit/count_signal
        if hitRate == 1: hitRate = 1-halfHit
        if hitRate == 0: hitRate = halfHit

        # Calculate false alarm rate, avoiding d' infinity
        faRate = count_fa/count_noise
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


def calc_roc(data, coding_dict=None):

    """Calculate ROC curve for a pandas dataframe
    Parameters
    ----------
    data : Pandas dataframe
        dataframe including cols for subject, objective status of each trial (e.g., signal/old, noise/new),
        response for each trial (e.g., 1-5 confidence scale)
    coding_dict : dict
        dictionary with information about objective (objective_col; string) and
        response columns (response_col; string), subject ID column (subj_col; string),
        objective signal (signal; list of strings) and noise (noise; list of strings) labels,
        and subjective responses (signal_resp; list of strings).
            Example:
            coding_dict = dict(objective_col='TrialType', # column name for new
                               signal=['old'], # objectively old label
                               noise=['similar', 'new'], # objectively new label
                               response_col='Response',
                               subj_col='Subject',
                               )

    Returns
    -------
    df : Pandas dataframe
        A longform dataframe with a column for subject ID, level of confidence, and
        proportions of responses for old and new trials
    """

    # get relevant info
    subj_col = coding_dict['subj_col']
    obj_col = coding_dict['objective_col']
    resp_col = coding_dict['response_col']
    signal = coding_dict['old']
    noise = coding_dict['new']

    max_resp = int(data[resp_col].max())

    # init new dataframe
    df = pd.DataFrame(columns=[subj_col, 'conf_level', 'signal', 'noise'])

    # calculate dprime for each subj
    for subj in data[subj_col].unique():

        data_s = data[data[subj_col] == subj]
        count_signal = data_s[data_s[obj_col].isin(signal)].Trial.count()
        count_noise = data_s[data_s[obj_col].isin(noise)].Trial.count()

        for level in range(1, max_resp+1):
            count_signal_tolevel = data_s[(data_s[obj_col].isin(signal)) &
                                       (data_s[resp_col] >= level)].Trial.count()
            count_noise_tolevel = data_s[(data_s[obj_col].isin(noise)) &
                                       (data_s[resp_col] >= level)].Trial.count()

            row = pd.Series({subj_col: subj,
                            'conf_level': level,
                            'signal': count_signal_tolevel/count_old,
                            'noise': count_noise_tolevel/count_new})
            df = df.append(row, ignore_index=True)

    return df
