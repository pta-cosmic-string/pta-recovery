import numpy as np
from .const import *

def make_freqs(Tspan, nfreq):
    freq = np.arange(1, nfreq + 1) / (Tspan)
    return freq

def make_dfreq(freqs):
    df = np.empty_like(freqs)
    df[0] = freqs[0]
    df[1:] = np.diff(freqs)
    return df

def residual_psd(freqs, h_c):
    psd = h_c**2 / (12 * np.pi**2 * freqs**3)
    return psd

def power_spec(freqs, A, gamma):
    alpha = (3.0 - gamma) / 2.0
    h_c = A * (freqs / F_YEAR) ** alpha
    return h_c 

def delta_spec(freq, A, f_0):
    index_f = np.argmin(abs(freq - f_0))
    h_c = np.zeros(len(freq))
    h_c[index_f] = A
    return h_c 

f_specs = {
    'pow' : power_spec,
    'delta': delta_spec,
}