# Copyright (c) 2015-2018 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["convergencetest"]

import numpy as np


def convergetest(chains, method='v21'):
    """
    Wrapper for the convergence test on a MCMC chain parameters.

    Parameters
    ----------
    chains : ndarray
        A 3D array of shape (nchains, nparameters, chainlen) containing
        the parameter MCMC chains.
    method : string
        Specification of which convergence test criterion to use.
        Options: gr92 - method of Gelman & Rubin (1992)
                 v21  - (default) method of Vehtari et al. (2021)

    Returns
    -------
    psrf : ndarray
        The potential scale reduction factors of the chain.  If the
        chain has converged, each value should be close to unity.  If
        they are much greater than 1, the chain has not converged and
        requires more samples.  The order of psrfs in this vector are
        in the order of the free parameters.

    Previous (uncredited) developers
    --------------------------------
    Chris Campo
    """
    # Allocate placeholder for results:
    npars = np.shape(chains)[1]
    psrf = np.zeros(npars)

    # Calculate psrf for each parameter:
    for i in range(npars):
      if method == 'gr92':
          psrf[i] = gelmanrubin(chains[:, i, :])
      elif method == 'v21':
          psrf[i] = vehtari(chains[:, i, :])
    return psrf


def gelmanrubin(chains):
    """
    Calculate the potential scale reduction factor of the Gelman & Rubin
    convergence test on a fitting parameter

    Parameters
    ----------
    chains: 2D ndarray
       Array containing the chains for a single parameter.  Shape
       must be (nchains, chainlen)
    """
    # Get length of each chain and reshape:
    nchains, chainlen = np.shape(chains)

    # Calculate W (within-chain variance):
    W = np.mean(np.var(chains, axis=1))

    # Calculate B (between-chain variance):
    means = np.mean(chains, axis=1)
    mmean = np.mean(means)
    B     = (chainlen/(nchains-1.0)) * np.sum((means-mmean)**2)

    # Calculate V (posterior marginal variance):
    V = W*((chainlen - 1.0)/chainlen) + B*((nchains + 1.0)/(chainlen*nchains))

    # Calculate potential scale reduction factor (PSRF):
    psrf = np.sqrt(V/W)

    return psrf


def vehtari(chains):
    """
    Calculate the potential scale reduction factor of the Vehtari et al.
    convergence test on a fitting parameter

    Parameters
    ----------
    chains: 2D ndarray
       Array containing the chains for a single parameter.  Shape
       must be (nchains, chainlen)
    """
    # Split chains in half, ensuring equal number of steps in each
    c1 = chains[:, chains.shape[-1]%2 : chains.shape[-1]//2 + chains.shape[-1]%2]
    c2 = chains[:, chains.shape[-1]%2 + chains.shape[-1]//2 :]

    # Stack them to have 2*nchains for split-Rhat (see note after Eqn 4)
    c    = np.concatenate((c1, c2))
    M, N = c.shape # nchains, nsamples
    S    = M * N   # total samples

    # Compute variances & Rhat, Eqns 1--4
    means = c.mean(axis=-1)
    mmean = means.mean()
    B     = N / (M - 1) * np.sum((means - mmean)**2)
    W     = np.sum(np.sum((c - means[:, None])**2, axis=-1)) / (N - 1) / M
    var   = (W * (N - 1) + B) / N
    Rhat  = np.sqrt(var / W)

    return Rhat


