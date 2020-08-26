from scipy.spatial.distance import braycurtis
import numpy as np
import pandas as pd
import math

def BrayCurtis(x, y):
    #return np.sum(np.abs(x - y)) / np.sum(x + y)
    return braycurtis(x, y)

def BrayCurtis_neutrality(x):
    # all species have equal abundance in neutral community
    neutral = np.ones_like(x)
    neutral /= np.sum(neutral)

    return braycurtis(neutral, x)

def KullbackLeibler(x, y, verbose=False):
    normx = x / np.sum(x)
    normy = y / np.sum(y)

    y[y == 0] = 1e-5
    #if np.any(y == 0):
    #    print("y is zero")
    #    return 0

    if verbose:
        print('normx', normx)
        print('normy', normy)
        print('normx/normy', normx / normy)
        print('log(normx/normy)', np.log(normx/normy))

    KL = np.sum(np.where(normy != 0, normx * np.log(normx / normy), 0))
    # use normx * log(normx / normy) values unless normy = 0, then use 0
    if KL < -1e-6:
        print("NEGATIVE KULLBACK LEIBLER!")
    elif KL < 0:
        return 0
    else:
        return KL

def KullbackLeibler_neutrality(y_ts, verbose=False):
    if isinstance(y_ts, pd.DataFrame):
        cols = [col for col in y_ts.columns if col.startswith('species')]
        y_ts = y_ts[cols].values

    if np.all(y_ts == y_ts[0, 0]):
        if verbose:
            print("All species are equal")
        return np.nan

    S = len(y_ts[0])
    mean = np.mean(y_ts, axis=0)
    mean_N = np.mean(mean)

    # x bar = J/S

    cov = np.zeros([S, S])

    for i in range(S):
        for j in range(S):
            cov[i, j] = np.mean((y_ts[:, i] - mean[i]) * (y_ts[:, j] - mean[j]))

    cov_diag0 = np.copy(cov);
    np.fill_diagonal(cov_diag0, 0)
    cov_N = np.sum(cov_diag0) / S / (S - 1) * np.ones([S, S])
    np.fill_diagonal(cov_N, np.mean(np.diag(cov)))

    trace_term = np.trace(np.dot(np.linalg.inv(cov_N), cov))
    mean_term = (((mean_N - mean).T).dot(np.linalg.inv(cov_N))).dot(mean_N - mean)
    rank_term = - np.linalg.matrix_rank(cov)
    # Shermann Morisson
    d = np.mean(np.diag(cov))  # diagonal term
    o = np.sum(cov_diag0) / S / (S - 1)  # offdiagonal term

    if np.log(d - o) * S < 700:
        det_cov_N = (d - o) ** S * (1 + float(S) * float(o) / float(d - o))
        log_det_cov_N = np.log(det_cov_N)
    else: # overflow
        log_det_cov_N = np.log(d - o) * S + np.log((1 + float(S) * float(o) / float(d - o)))
        det_cov_N = np.exp(log_det_cov_N)

    if np.isnan(log_det_cov_N):
        if verbose:
            print("overflow (d = %.3E, o = %.3E" % (d, o))
        return np.nan

    det_cov = np.linalg.det(cov)

    if det_cov == 0 or det_cov_N == 0:
        determinant_term = (S * (np.log(np.mean(np.abs(cov_N))) - np.log(np.mean(np.abs(cov)))) + np.log(
            np.linalg.det(cov_N / np.mean(np.abs(cov_N)))) - np.log(np.linalg.det(cov / np.mean(np.abs(cov)))))
    else:
        determinant_term = log_det_cov_N - np.log(np.linalg.det(cov))

    KL = 1 / 2 * (trace_term + mean_term + rank_term + determinant_term)

    if verbose:
        eig = np.linalg.eigvals(cov);
        eig = eig[eig != 0];

        log_pseudodet = np.sum(np.log(eig))
        if log_pseudodet > 700:
            return np.nan
        else:
            pseudodet = np.exp(log_pseudodet)

        eig = np.linalg.eigvals(cov_N);
        eig = eig[eig != 0];

        log_pseudodet_N = np.sum(np.log(eig))
        pseudodet_N = np.exp(log_pseudodet)

        print("cov", cov[:4, :4])
        print("trace", trace_term)
        print("rank", rank_term)
        print("mean", mean_term)
        print("determinant_term", determinant_term)
        print("determinants scaled (neutral + original)", np.linalg.det(cov_N / np.mean(np.abs(cov_N))),
              np.linalg.det(cov / np.mean(np.abs(cov))))
        print("determinants (neutral + original)", np.linalg.det(cov_N), np.linalg.det(cov))
        print("determinant neutral Shermann Morisson", det_cov_N)
        print("pseudodeterminants (neutral + niche)", pseudodet_N, pseudodet)
        print("Kullback Leibler", KL)

        eig = np.linalg.eigvals(cov)

        print("eig", eig)
    return KL

def JensenShannon(x, y, verbose=False):
    if verbose:
        print("Kullback Leibler x and (x+y)/2:", KullbackLeibler(x, (x + y) / 2.0, verbose))
        print("Kullback Leibler y and (x+y)/2:", KullbackLeibler(y, (x + y) / 2.0, verbose))
    return np.sqrt(0.5 * KullbackLeibler(x, (x + y) / 2.0) + 0.5 * KullbackLeibler(y, (x + y) / 2.0))

def test_compare_braycurtis_definitions():
    x = np.random.uniform(0,10,10)
    y = np.random.uniform(0,10,10)

    bc1 = braycurtis(x, y)
    bc2 = BrayCurtis(x, y)
    print("The Bray-Curtis distance of the scipy.spatial.distance package is:", bc1)
    print("The Bray-Curtis distance ( sum(abs(x-y)) / sum(x+y) ) is:", bc2)
    print("The difference between both definitions is ", bc1 - bc2)

def Pielou_evenness_index(x):
    p = x / np.sum(x)
    HShannon = - np.sum(np.log(p ** p))
    Hmax = np.log(len(x))
    return HShannon/Hmax