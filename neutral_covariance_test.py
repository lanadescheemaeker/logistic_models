# translation of the original Washburne code in R

import scipy.stats
from scipy.special import kolmogorov
import numpy as np
from statsmodels.stats.diagnostic import het_breuschpagan
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smfa
from multiprocessing import Pool
from functools import partial

# TODO not sure if this file is correct, small numerical deviations

def predict(ntests, S, qr=False):
    aa = -1
    bb = 3

    return (aa + bb * np.log(S) - np.log(ntests))

def draw_WFP_CVT(ts, n=None, m=None):
    # constant-volatility transformation

    if n == None:
        n = ts.shape[1] # number of species
    if m == None:
        m = ts.shape[0] # number of timesteps

    #dum = True
    #while dum == True:
    # array of random 1's and -1's
    a = np.random.uniform(-1,1,[n, 1]); a[a<0] = -1; a[a>0] = 1;
    #   if (abs(sum(a) < n)):
    #       dum = False

    vec = ts.dot(a).flatten()
    vec[vec < -1] = -1
    vec[vec > 1] = 1
    output = pd.DataFrame(columns=['f', 'DF'])
    output['f'] = np.arcsin(vec[1:])
    output['DF'] = np.arcsin(vec[1:]) - np.arcsin(vec[:-1])

    return output

def pval(ts, regress='f', formula=None, DT=0, seed=0):
    np.random.seed(seed)

    cvt = draw_WFP_CVT(ts)
    cvt['f2'] = cvt['f']**2

    if not regress == 'f':
        cvt['DF'] = cvt['DF'] / np.sqrt(DT)

    # linear fit as in formula
    try:
        model = smfa.ols(formula=formula, data=cvt).fit()
    except:
        print("code was failing, cvt")
        print(cvt)
    _, r, _, _ = het_breuschpagan(model.resid, model.model.exog) # pvalue chi-squared test _, _, _, pvalue

    return r

def cv_test(ntests, ts, regress='f', formula=None, varformula=None, tm=None, ncores=1, seed=0):
    np.random.seed(seed)

    if not regress in ['f', 'time']:
        print('Error: unknown inputt "regress" - must be either "f" or "time"')
        return

    if formula == None:
        if regress == 'f':
            formula = 'DF~f+f2'
        else:
            formula = 'DF~tm'

    if varformula == None:
        if regress == 'f':
            varformula = 'DF~f+f2'
        else:
            varformula = 'DF~tm+tm2'

    p = np.zeros(ntests)
    b = np.zeros([3, ntests]) # intercept, coef for f, coef for f**2

    DT = 0
    if regress == 'time' and tm == None:
        tm = np.arange(1, len(ts))
        DT = tm[1:] - tm[:-1]
        tm = tm[:-1]

    if ncores == 1:
        for nn in range(ntests):
            p[nn] = pval(ts, regress, formula, seed=np.random.randint(1e6))

            if False:
                if regress == 'f':
                    cvt['DF'] = model.resid ** 2

                b[:, nn] = sm.formula.glm(varformula, data=cvt).fit().params
                # orig code glm function -> iteratively reweighted least squares (IWLS)
                # can give rise to an error: PerfectSeparationError: Perfect separation detected, results not available
    else:
        pval_ts = partial(pval, ts, regress, formula, DT)

        pool = Pool(ncores)
        p = pool.map(pval_ts, np.random.randint(0,1e6,ntests))

    return p

def neutral_covariance_test(ts, ntests=None, regress='f', formula='DF~f+I(f**2)',
                            varformula=None, standard=True, method='logitnorm', verbose=False, ncores=1, seed=0):

    if method not in ['Kolmogorov', 'logitnorm', 'uncorrected']:
        print('Unknown input method. Must be either "Kolmogorov", "logitnorm", or "uncorrected".')
        return

    S = ts.shape[1]  # number of species
    m = ts.shape[0]  # number of timepoints
    print("CVT test: number of timepoints: %d, number of species: %d" % (m, S))

    # check if timeseries are normalized
    sum_ts = np.sum(ts, axis=1)
    if max(sum_ts) > 1.01 or min(sum_ts) < 0.99:
        raise ValueError('Timeseries is not normalized.')

    upperbound = 0.2889705

    if ntests == None:
        ntests = min(S, int((upperbound * S) ** 3))
    elif ntests > (upperbound * S) ** 3:
        print('Warning: ntests input is large relative to number of species, leading to high false-positive rates for P<0.05')

    pvalues = cv_test(ntests, ts, regress, formula, varformula, ncores=ncores, seed=seed)

    if verbose:
        print("pvals", pvalues)

    ntests = len(pvalues)
    D = scipy.stats.kstest(pvalues, 'uniform').statistic

    if verbose:
        print("D", D)

    if method == 'Kolmogorov':
        Q = predict(ntests, S)
        if verbose:
            print("Q", Q)
        nstar_est = ntests / (1.0 + np.exp(-Q))
        if verbose:
            print("nstar_est", nstar_est)
        P = kolmogorov(D * np.sqrt(nstar_est))  # complementary cumulative Kolmogorov distribution

    return P

def normalize(ts):
    ts /= np.sum(ts, axis=1, keepdims=True)
    return ts


def main():
    #ts = np.loadtxt('../../enterotypes_ibm/timeseries-Faust/5_timeseries/5_timeseries.txt', skiprows=1).T
    #ts = pd.read_csv('../study_with_interaction/timeseries_Langevin_linear_interaction0.csv', index_col=1).values
    ts = pd.read_csv('../study_no_interaction/timeseries_Langevin_linear_dt1.csv', index_col=1).values
    ts = normalize(ts)
    print(neutral_covariance_test(ts, ntests=500, method = 'Kolmogorov', verbose=False))

if __name__ == "__main__":
    main()