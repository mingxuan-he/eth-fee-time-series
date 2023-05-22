"""
This file implements two functions used for time series tests:
Ljung_Box_Q_test and Check_Stationarity
"""

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt


def autocov(z, j):
    """
    autocovariance function of j-th order
    z: data series (array)
    j: order
    """
    n = len(z)
    z_mean = np.mean(z)

    gamma = 0.0
    for t in range(j,n):
        gamma += (z[t] - z_mean) * (z[t-j] - z_mean)
    gamma = gamma/n

    return gamma


def autocorr(z, j):
    """"
    autocorrelation function of j-th order
    z: data series (array)
    j: order
    """
    gamma_0 = autocov(z,0)
    gamma_j = autocov(z,j)
    rho = gamma_j / gamma_0

    return rho


# Correlogram
def correlogram(z, max_lag=40, plot=False):
    # z: data series
    # max_lag: number of lags
    correlogram = []
    for j in range(max_lag+1):
        rho_j = autocorr(z,j)
        correlogram.append(rho_j)

    # stderr
    N = len(z)
    stderr = 1 / np.sqrt(N)

    # plot correlogram
    if plot:
        plt.figure(figsize=(20,8))
        plt.plot(range(max_lag+1), correlogram, "o-")
        plt.axhline(2*stderr, ls="--", label='two std errs')
        plt.axhline(-2*stderr, ls="--")

        plt.title("Correlogram", fontsize=20)
        plt.xlabel("Lag")
        plt.ylabel("sample correlation coefficient")
        plt.legend(fontsize=20)
        plt.show()
    
    return correlogram, stderr


def Ljung_Box_Q_test(z,j):
    """
    Computes the Ljung-Box Q statistic 
        and the p-value basd on the Q test
    See reference: Fumio Hayashi (2011) Econometrics Ch. 2.10
    z: data series (array)
    j: order
    """
    n = len(z)
    q = 0.0
    for i in range(1,j+1):
        q += autocorr(z,i)**2 / (n-i)
    q = q * n *(n+2)

    # Q-stat asympototically converges to chi-square distribution
    p = 1 - sts.chi2.cdf(q, j)

    return q, p


from statsmodels.tsa.stattools import adfuller

def Check_Stationarity(z):
    """
    Performs Augmented Dickey-Fuller's test
    z: data series (array)
    """
    
    test_result = adfuller(z)
    print("ADF Statistic : %f \n" %test_result[0])
    print("p-value : %f \n" %test_result[1])
    print("Critical values are: \n")
    print(test_result[4])

    return test_result