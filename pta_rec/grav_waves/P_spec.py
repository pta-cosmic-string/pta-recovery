import numpy as np
import healpy as hp
from tqdm import tqdm
from .funcs import *

def get_npix(nside):
    return hp.nside2npix(nside)

def get_nside(npix):
    return hp.npix2nside(npix)

def get_Omega(nside):
    Omega = np.array(hp.pix2vec(nside, np.arange(get_npix(nside))))
    return Omega.T

def get_dOmega(nside):
    dOmega = hp.nside2pixarea(nside)
    return dOmega

def get_Y_lm(nside, l, m, nside_max=1024):
    lmax = 3 * nside_max - 1
    alm_size = (lmax + 1) * (lmax + 2) // 2
    idx = hp.sphtfunc.Alm.getidx(lmax, l, abs(m))
    if m==0:
        alm = np.zeros(alm_size, dtype=np.complex128)
        alm[idx] = 1.0
        map_lm = hp.alm2map(alm, nside, lmax=lmax)
    else:
        alm = np.zeros(alm_size, dtype=np.complex128)
        alm[idx] = 1.0
        map_re = 0.5 * hp.alm2map(alm, nside, lmax=lmax)

        alm = np.zeros(alm_size, dtype=np.complex128)
        alm[idx] = 1.0j
        map_im = - 0.5 * hp.alm2map(alm, nside, lmax=lmax)

        map_lm = map_re + 1j * map_im

        if m<0:
            map_lm = (-1)**m * np.conjugate(map_lm)

    return map_lm

def hellings_downs_curve(theta):
    x = np.cos(theta)
    if np.isclose(theta, 0.0):
        return 1.0
    return 1.0 + 3/2 * (1 - x) * (np.log((1 - x)/2) - 1/6)

def hellings_downs_kernel(p1, p2, Omega_0):
    a = np.clip(np.dot(Omega_0, p1), -1.0, 1.0)
    b = np.clip(np.dot(Omega_0, p2), -1.0, 1.0)
    c = np.clip(np.dot(p1, p2), -1.0, 1.0)
    return 3/4 * (
        2*(c - a * b)**2/((1 + a)*(1 + b))  - (1 - a)*(1 - b)
    ) 

def spherical_harmonic(nside, l, m):
    Y_lm = get_Y_lm(nside, l, abs(m))
    if m==0:
        map_data = np.real(Y_lm)
    elif m>0:
        map_data = np.sqrt(2) * np.real(Y_lm)
    else:
        map_data = np.sqrt(2) * np.imag(Y_lm)

    return map_data

def generate_power_map(nside, alpha=1, map_seed=42, nside_max=1024):
    np.random.seed(map_seed)
    dOmega = get_dOmega(nside)
    lmax = 3*nside - 1
    lmax_fixed = 3*nside_max-1

    ell = np.arange(lmax + 1, dtype=float)
    Cl = np.zeros_like(ell)
    mask = ell >= 1
    C_0, C_1 = 1.0, 0.4
    Cl[mask] = C_1/C_0 * ell[mask] ** (-alpha)   # C_ℓ = ℓ^{-α}
    Cl[0] = 1.0

    map_data = hp.synfast(Cl, nside, lmax=lmax_fixed)
    p_data = np.abs(map_data)**2
    norm = np.sum(p_data * dOmega)
    p_data_norm = p_data/norm

    return p_data_norm

def generate_phisical_map(nside, beta=2, ell_break=10, map_seed=42, nside_max=1024):
    np.random.seed(map_seed)
    dOmega = get_dOmega(nside)
    lmax = 3*nside - 1
    lmax_fixed = 3*nside_max-1

    C_0, C_1 = 1.0, 0.4
    A = C_1 * (1 + ell_break**(-beta))

    ell = np.arange(lmax + 1, dtype=float)
    Cl = np.zeros_like(ell)
    mask = ell >= 1
    Cl[mask] = A * 1/ell[mask] * 1 / (1 + (ell[mask]/ell_break)**beta) 

    map_data = hp.synfast(Cl, nside, lmax=lmax_fixed)
    p_data = np.abs(map_data)**2
    norm = np.sum(p_data * dOmega)
    p_data_norm = p_data/norm

    return p_data_norm
    
def isotropic_hd(p1, p2):
    x = np.clip(np.dot(p1, p2), -1.0, 1.0)
    if np.isclose(angular_distance(p1, p2), 0.0):
        return 1.0
    return 1.0 + 3/2 * (1 - x) * (np.log((1 - x)/2) - 1/6)

def point_hd(p1, p2, Omega_0):
    a = np.clip(np.dot(Omega_0, p1), -1.0, 1.0)
    b = np.clip(np.dot(Omega_0, p2), -1.0, 1.0)
    c = np.clip(np.dot(p1, p2), -1.0, 1.0)
    return 3/4 * (
        2*(c - a * b)**2/((1 + a)*(1 + b))  - (1 - a)*(1 - b)
    ) 

def gaussian_hd(p1, p2, Omega_0, kappa, eps=1e-10):
    a = np.clip(np.dot(Omega_0, p1), -1.0 + eps, 1.0 - eps)
    b = np.clip(np.dot(Omega_0, p2), -1.0 + eps, 1.0 - eps)
    c = np.clip(np.dot(p1, p2), -1.0 + eps, 1.0 - eps)
    k = kappa

    V2 = 1.0 + 2.0*a*b*c - c*c - a*a - b*b
    V = np.sqrt(max(0.0, V2))
    t = (a + b)/(1.0 + c)
    s = V/(1.0 + c)

    coth = 1.0/np.tanh(k)
    chc_2 = coth - 1.0/k
    chc_3 = 3.0*coth/k - 3.0/k**2 - 1.0
    chc_a = chc_endpoint(a, k)
    chc_b = chc_endpoint(b, k)

    one_minus_a2 = max(1e-12, 1.0 - a*a)
    one_minus_b2 = max(1e-12, 1.0 - b*b)

    return 3.0 * (
        1.0/3.0
        + 0.5 * (
            c * (chc_a + chc_b - chc_3/3.0)
            + (a*b - c) * (chc_a/one_minus_a2 + chc_b/one_minus_b2 - chc_3/2.0)
            - (a + b) * (chc_2/2.0)
        )
        + 0.5 * (1.0 - c) * (
            + expi_stable(t, s, a, k)
            + expi_stable(t, s, b, k)
            - expi_stable(t, s, -1.0, k)
            - expi_stable(t, s, +1.0, k)
            - 1.0/6.0
        )
    )

def numeric_hd(p1, p2, P_spec=None):
    nside = get_nside(len(P_spec))
    Omega, dOmega = get_Omega(nside), get_dOmega(nside)
    K = hellings_downs_kernel(p1, p2, Omega)
    return np.sum(K * P_spec * dOmega)

def isotropic_spec(Omega):
    Nomega = len(Omega)
    P = np.ones(Nomega)
    norm = 4 * PI
    return (1 / norm) * P 

def point_spec(Omega, Omega_0):
    Nomega = len(Omega)
    P = np.zeros(Nomega)
    index_P = np.argmin(angular_distance(Omega, Omega_0))
    P[index_P] = 1.0
    norm = 4 * PI / Nomega
    return (1 / norm) * P 

def gaussian_spec(Omega, Omega_0, kappa):
    delta_Omega = np.clip(np.dot(Omega, Omega_0), -1.0, 1.0)
    P = np.exp(kappa * (delta_Omega-1))
    norm = 4 * PI * (1 - np.exp(-2*kappa)) / (2*kappa)
    return (1 / norm) * P 

def numeric_spec(Omega, P_spec=None):
    P = P_spec
    norm = 1.0
    return (1 / norm) * P

def calc_ORF(pulsars, hellings_downs, params):
    Npsr = len(pulsars)
    Gamma = np.zeros((Npsr, Npsr), dtype=float)
    for i in tqdm(range(Npsr)):
        for j in range(Npsr):
            p1, p2 = to_vec(pulsars[i]), to_vec(pulsars[j])
            Gamma[i, j] = hellings_downs(p1, p2, **params)
    return Gamma

def eval_ORF(pulsars, ngrid=1000):
    Npsr = len(pulsars)
    Сov = np.zeros((Npsr, Npsr), dtype=float)
    t_grid = get_common_grid(pulsars, ngrid)
    interp_res = [interpolate_residuals(p, t_grid) for p in pulsars]
    for i in range(Npsr):
        for j in range(Npsr):
            Сov[i, j] = covariance(interp_res[i], interp_res[j])
    sigma2 = np.mean(np.diag(Сov)) 
    Gamma = Сov/sigma2
    return Gamma

P_specs = {
    'iso': {
        'P_spec': isotropic_spec,
        'hd': isotropic_hd,
    },
    'point': {
        'P_spec': point_spec,
        'hd': point_hd,
    },
    'gauss':{
        'P_spec': gaussian_spec,
        'hd': gaussian_hd,
    },
    'numeric': {
        'P_spec': numeric_spec,
        'hd': numeric_hd,
    }
}