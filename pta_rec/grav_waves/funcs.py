import numpy as np
import scipy as sp
from .const import *

def to_unit(ra, dec):
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])

def to_coord(vec):
    return hp.ang2vec(vec)

def to_vec(psr):
    u = to_unit(psr['RAJ'].val, psr['DECJ'].val)
    return u
    
def get_toas(psr):
    return np.array(psr.toas(), dtype=np.float64)

def get_residuals(psr):
    return np.array(psr.residuals(), dtype=np.float64)

def get_tspan(pulsars):
    t_min = max(np.min(get_toas(p)) for p in pulsars)
    t_max = min(np.max(get_toas(p)) for p in pulsars)
    return (t_max - t_min) * DAY

def get_common_grid(pulsars, ngrid=200):
    t_min = max(np.min(get_toas(p)) for p in pulsars)
    t_max = min(np.max(get_toas(p)) for p in pulsars)
    return np.linspace(t_min, t_max, ngrid)

def angular_distance(u1, u2):
    cos_theta = np.clip(np.dot(u1, u2), -1.0, 1.0)
    return np.arccos(cos_theta)

def interpolate_residuals(pulsars, t_grid):
    t = get_toas(pulsars)
    r = get_residuals(pulsars)

    idx = np.argsort(t)
    t = t[idx]
    r = r[idx]

    t_unique, unique_idx = np.unique(t, return_index=True)
    r = r[unique_idx]

    return np.interp(t_grid, t_unique, r)

def correlation(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)

    denom = np.std(x) * np.std(y)
    if denom == 0:
        return 0.0

    return np.mean(x * y) / denom

def covariance(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    return np.mean(x * y)   # или np.cov(x, y, ddof=1)[0,1]

def average(gamma, mu, n_average = 12):
    dgamma = PI / n_average
    gamma_mean = np.linspace(dgamma/2, PI - dgamma/2, n_average)

    mu_m = np.zeros(n_average)
    mu_m2 = np.zeros(n_average) 
    N_m = np.zeros(n_average)

    for k in range(gamma.shape[0]):
        Gamma_k, gamma_k = mu[k], gamma[k]
        mask = (gamma_mean - dgamma <= gamma_k) & (gamma_k < gamma_mean + dgamma)
        mu_m[mask] += Gamma_k
        mu_m2[mask] += Gamma_k**2
        N_m[mask] += 1

    mu_mean = mu_m / N_m
    mu_std = np.sqrt(mu_m2 / N_m - mu_mean**2) 

    return gamma_mean, mu_mean, mu_std


def scaled_expi(w, switch=100.0, max_terms=50):
    w = np.asarray(w, dtype=np.complex128)

    if w.ndim == 0:
        if abs(w) < switch:
            return - np.exp(-w) * sp.special.exp1(-w)
        term = 1.0 / w
        s = term
        for n in range(1, max_terms):
            term *= n / w
            s_new = s + term
            if abs(term) <= np.finfo(float).eps * abs(s_new):
                return s_new
            s = s_new
        return s

    out = np.empty_like(w)
    small = np.abs(w) < switch
    out[small] = - np.exp(-w[small]) * sp.special.exp1(-w[small])

    big = ~small
    if np.any(big):
        wb = w[big]
        term = 1.0 / wb
        s = term.copy()
        for n in range(1, max_terms):
            term *= n / wb
            s_new = s + term
            if np.all(np.abs(term) <= np.finfo(float).eps * np.abs(s_new)):
                s = s_new
                break
            s = s_new
        out[big] = s

    return out

def expi_stable(x, s, a, k):
    z = x + 1j * s
    w = k * (z - a)
    norm = 2.0 * k * np.exp(-k * (a + 1.0)) / (-np.expm1(-2.0 * k))
    return norm * np.real(scaled_expi(w))

def exp_minus_kmu_over_sinh_k(mu, k):
    return 2.0 * np.exp(-k * (mu + 1.0)) / (-np.expm1(-2.0 * k))

def chc_endpoint(mu, k):
    if abs(mu) < 1e-7:
        return 1.0 / np.tanh(k) - 1.0 / k
    ratio = exp_minus_kmu_over_sinh_k(mu, k)
    return 1.0 / (np.tanh(k) * mu) - ratio / mu - 1.0
