import numpy as np
from .const import *

def gwb_coefficients_base(orf, freqs, psd, seed):
    rng = np.random.default_rng(seed)
    Npsr, Nf = len(orf), len(freqs)
    df = np.empty_like(freqs)
    df[0] = freqs[0]
    df[1:] = np.diff(freqs)

    a = np.zeros((Npsr, Nf))
    b = np.zeros((Npsr, Nf))

    for n in range(Nf):
        Cn = orf * psd[n] * df[n]
        L = np.linalg.cholesky(Cn)
        a[:, n] = L @ rng.normal(size=Npsr)
        b[:, n] = L @ rng.normal(size=Npsr)

    return a, b


def gwb_coefficients(orf, freqs, df, psd, seed):
    rng = np.random.default_rng(seed)
    Npsr, Nf = len(orf), len(freqs)
    a = np.zeros((Npsr, Nf))
    b = np.zeros((Npsr, Nf))

    for n in range(Nf):
        Cn = orf * psd[n] * df[n]
        # Гарантируем симметричность (на всякий случай)
        Cn = (Cn + Cn.T) / 2.0

        # Разложение по собственным значениям
        eigvals, eigvecs = np.linalg.eigh(Cn)
        # Обрезаем отрицательные значения (из-за ошибок)
        eigvals = np.maximum(eigvals, 0.0)
        # Строим L = V * sqrt(diag(eigvals))
        L = eigvecs @ np.diag(np.sqrt(eigvals))

        a[:, n] = L @ rng.normal(size=Npsr)
        b[:, n] = L @ rng.normal(size=Npsr)

    return a, b

def fourier_design_matrix(toas, freqs):
    phase = 2 * PI * np.outer(toas, freqs)
    F = np.empty((len(toas), 2 * len(freqs)))
    F[:, 0::2] = np.cos(phase)
    F[:, 1::2] = np.sin(phase)
    return F