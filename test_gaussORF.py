from __future__ import division

import os

# fixed: set thread limits for the current Python process
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
from itertools import combinations
import copy
import json
import sys

import matplotlib
matplotlib.use("Agg")  # fixed: safe backend for server without display

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import libstempo as T
import libstempo.plot as LP
import toasim as LT
from libstempo.libstempo import GWB

import dynesty
import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals.parameter import function
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise_extensions import model_utils, blocks

import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import scipy.interpolate as interp


def cart2sph(x, y, z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)
    elev = np.arctan2(z, np.sqrt(XsqPlusYsq))
    az = np.arctan2(y, x) + np.pi
    return r, elev, az


def cos_th(th1, ph1, th2, ph2):
    cth = np.sin(th1) * np.sin(th2) * np.cos(ph1 - ph2) + np.cos(th1) * np.cos(th2)
    return cth


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def ang_dist(ra1, ra2, dec1, dec2):
    return np.arccos(
        np.sin(dec1) * np.sin(dec2) +
        np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )


def fibonacci_sphere(num_points: int):
    ga = (3 - np.sqrt(5)) * np.pi

    phi = ga * np.arange(num_points)
    z = np.linspace(1 / num_points - 1, 1 - 1 / num_points, num_points)
    radius = np.sqrt(1 - z * z)

    y = radius * np.sin(phi)
    x = radius * np.cos(phi)

    r, theta, phi = cart2sph(x, y, z)
    return theta, phi




###functions for ORF
###############################################################################
def _gw_source_unit_vector(gwtheta, gwphi):
    """Source center unit vector for gwtheta=colatitude and gwphi=longitude."""
    return np.array([
        np.sin(gwtheta) * np.cos(gwphi),
        np.sin(gwtheta) * np.sin(gwphi),
        np.cos(gwtheta),
    ], dtype=float)


def _scaled_expi(w, switch=100.0, max_terms=50):
    """Stable exp(-w) * Ei(w)."""
    w = np.asarray(w, dtype=np.complex128)

    if w.ndim == 0:
        if abs(w) < switch:
            return np.exp(-w) * sp.special.expi(w)
        term = 1.0 / w
        out = term
        for n in range(1, max_terms):
            term *= n / w
            new = out + term
            if abs(term) <= np.finfo(float).eps * max(1.0, abs(new)):
                return new
            out = new
        return out

    out = np.empty_like(w)
    small = np.abs(w) < switch
    out[small] = np.exp(-w[small]) * sp.special.expi(w[small])
    big = ~small
    if np.any(big):
        wb = w[big]
        term = 1.0 / wb
        ss = term.copy()
        for n in range(1, max_terms):
            term *= n / wb
            new = ss + term
            if np.all(np.abs(term) <= np.finfo(float).eps * np.maximum(1.0, np.abs(new))):
                ss = new
                break
            ss = new
        out[big] = ss
    return out


def _expi_stable(x, s, a, k):
    z = x + 1j * s
    w = k * (z - a)
    norm = 2.0 * k * np.exp(-k * (a + 1.0)) / (-np.expm1(-2.0 * k))
    return norm * np.real(_scaled_expi(w))


def _exp_minus_kmu_over_sinh_k(mu, k):
    """Stable exp(-k*mu) / sinh(k)."""
    return 2.0 * np.exp(-k * (mu + 1.0)) / (-np.expm1(-2.0 * k))


def _chc_endpoint(mu, k):
    """coth(k)/mu - exp(-k*mu)/(mu*sinh(k)) - 1, stable near mu=0."""
    if abs(mu) < 1e-7:
        # limit: coth(k) - 1/k
        return 1.0 / np.tanh(k) - 1.0 / k

    ratio = _exp_minus_kmu_over_sinh_k(mu, k)
    return 1.0 / (np.tanh(k) * mu) - ratio / mu - 1.0


def K_exp(Omega, p1, p2, kappa):
    Omega = np.asarray(Omega, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    Omega = Omega / np.linalg.norm(Omega)
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)

    a = float(np.clip(np.dot(Omega, p1), -1.0 + 1e-12, 1.0 - 1e-12))
    b = float(np.clip(np.dot(Omega, p2), -1.0 + 1e-12, 1.0 - 1e-12))
    c = float(np.clip(np.dot(p1, p2), -1.0 + 1e-12, 1.0 - 1e-12))

    k = float(kappa)
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("kappa must be positive and finite")

    V2 = 1.0 + 2.0*a*b*c - c*c - a*a - b*b
    V = np.sqrt(max(0.0, V2))
    t = (a + b)/(1.0 + c)
    s = V/(1.0 + c)

    coth = 1.0/np.tanh(k)
    chc_2 = coth - 1.0/k
    chc_3 = 3.0*coth/k - 3.0/k**2 - 1.0
    chc_a = _chc_endpoint(a, k)
    chc_b = _chc_endpoint(b, k)

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
            _expi_stable(t, s, a, k)
            + _expi_stable(t, s, b, k)
            - _expi_stable(t, s, -1.0, k)
            - _expi_stable(t, s, +1.0, k)
            - 1.0/6.0
        )
    )
###############################################################################



KAPPA = float(sys.argv[1])
KAPPA_TAG = f"{KAPPA:g}"
RUN_NAME = f"open2_gwb_kappa{KAPPA_TAG}"

datadir_in = "./"
datadir_out = f"psrE_kappa{KAPPA_TAG}/"
outdir = f"{datadir_out}final"                                                                                                        ##############################
chains_dir = f"chains_kappa{KAPPA_TAG}/mdc/{RUN_NAME}"


prefix_psr = "J"
Npsr = 20
coord = "cone"

cap_angle =180 * np.pi / 180.0
ra0 = np.pi / 2
dec0 = 0

psrcat_cat = np.genfromtxt(datadir_in + "psrcat_data.txt", skip_header=1, dtype="str", unpack=True)
rand_n = np.random.choice(len(psrcat_cat.T), Npsr, replace=False)
num_cat, pmra_cat, pmdec_cat, px_cat, rajd_cat, decjd_cat, f0_cat, f1_cat, dm_cat = psrcat_cat


def radec_to_unit(ra, dec):
    """ra, dec (рад) -> единичный вектор (x,y,z)."""
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])


def unit_to_radec(v):
    """(N,3) массив единичных векторов -> ra, dec (рад)."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    ra = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
    dec = np.arcsin(np.clip(z, -1.0, 1.0))
    return ra, dec


def psrpos_from_radec(ra, dec):
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])


def make_gaussian_orf_matrix(psr, gwtheta, gwphi, kappa):
    Npulsars = len(psr)
    psrpos = []

    for ii in range(Npulsars):
        if "RAJ" in psr[ii].pars() and "DECJ" in psr[ii].pars():
            ra = np.double(psr[ii]["RAJ"].val)
            dec = np.double(psr[ii]["DECJ"].val)
            psrpos.append(psrpos_from_radec(ra, dec))
        else:
            raise ValueError("Gaussian ORF needs RAJ and DECJ for every pulsar")

    psrpos = np.array(psrpos)

    ORF = np.zeros((Npulsars, Npulsars))
    ang_dist_arr = np.zeros((Npulsars, Npulsars))

    for i in range(Npulsars):
        for j in range(i, Npulsars):
            val = K_exp(_gw_source_unit_vector(gwtheta, gwphi), psrpos[i], psrpos[j], kappa)
            ang_dist_arr[i,j] = np.dot(psrpos[i], psrpos[j])
            if not np.isfinite(val):
                val = 0.0

            ORF[i, j] = val
            ORF[j, i] = val

    ORF = 0.5 * (ORF + ORF.T)

    mineig = np.min(np.linalg.eigvalsh(ORF))
    if mineig <= 0:
        ORF += (abs(mineig) + 1e-6) * np.eye(Npulsars)
    ORF[np.abs(ORF)>10.] = 0.
    print(max(ORF.reshape(-1)))        
    plt.plot(ang_dist_arr.reshape(-1), ORF.reshape(-1), ".")
    plt.ylim(-2, 2)
    plt.savefig("orf.png", dpi=300)
    return ORF


def extrap1d(interpolator):
    """
    Function to extend an interpolation function to an
    extrapolation function.

    :param interpolator: scipy interp1d object

    :returns ufunclike: extension of function to extrapolation
    """

    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]  # +(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]  # +(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(list(map(pointwise, np.array(xs))))

    return ufunclike

def createGWB(
    psr,
    Amp,
    gam,
    gwtheta=None,
    gwphi=None,
    kappa=None,
    useGaussianORF=False,
    noCorr=False,
    seed=None,
    turnover=False,
    clm=[np.sqrt(4.0 * np.pi)],
    lmax=0,
    f0=1e-9,
    beta=1,
    power=1,
    cadence=14,
    userSpec=None,
    npts=600,
    howml=1,
):
    """
    Function to create GW-induced residuals from a stochastic GWB as defined
    in Chamberlin, Creighton, Demorest, et al. (2014).

    :param psr: pulsar object for single pulsar
    :param Amp: Amplitude of red noise in GW units
    :param gam: Red noise power law spectral index
    :param noCorr: Add red noise with no spatial correlations
    :param seed: Random number seed
    :param turnover: Produce spectrum with turnover at frequency f0
    :param clm: coefficients of spherical harmonic decomposition of GW power
    :param lmax: maximum multipole of GW power decomposition
    :param f0: Frequency of spectrum turnover
    :param beta: Spectral index of power spectram for f << f0
    :param power: Fudge factor for flatness of spectrum turnover
    :param userSpec: User-supplied characteristic strain spectrum
                     (first column is freqs, second is spectrum)
    :param npts: Number of points used in interpolation
    :param howml: Lowest frequency is 1/(howml * T)

    :returns: list of residuals for each pulsar
    """

    if seed is not None:
        np.random.seed(seed)

    # number of pulsars
    Npulsars = len(psr)

    # gw start and end times for entire data set
    start = np.min([p.toas().min() * 86400 for p in psr]) - 86400
    stop = np.max([p.toas().max() * 86400 for p in psr]) + 86400

    # duration of the signal
    dur = stop - start

    # get maximum number of points
    if npts is None:
        # default to cadence of 2 weeks
        npts = dur / (86400 * cadence)

    # make a vector of evenly sampled data points
    ut = np.linspace(start, stop, npts)

    # time resolution in days
    dt = dur / npts

    # compute the overlap reduction function
    if noCorr:
        ORF = np.diag(np.ones(Npulsars) * 2)

    elif useGaussianORF:
        if gwtheta is None or gwphi is None or kappa is None:
            raise ValueError("Need gwtheta, gwphi and kappa for Gaussian ORF")
        ORF = make_gaussian_orf_matrix(psr, gwtheta, gwphi, kappa)

    else:
        psrlocs = np.zeros((Npulsars, 2))

        for ii in range(Npulsars):
            if "RAJ" and "DECJ" in psr[ii].pars():
                psrlocs[ii] = np.double(psr[ii]["RAJ"].val), np.double(psr[ii]["DECJ"].val)
            elif "ELONG" and "ELAT" in psr[ii].pars():
                fac = 180.0 / np.pi
                # check for B name
                if "B" in psr[ii].name:
                    epoch = "1950"
                else:
                    epoch = "2000"
                coords = ephem.Equatorial(
                    ephem.Ecliptic(str(psr[ii]["ELONG"].val * fac), str(psr[ii]["ELAT"].val * fac)), epoch=epoch
                )
                psrlocs[ii] = float(repr(coords.ra)), float(repr(coords.dec))

        psrlocs[:, 1] = np.pi / 2.0 - psrlocs[:, 1]
        anisbasis = np.array(anis.CorrBasis(psrlocs, lmax))
        ORF = sum(clm[kk] * anisbasis[kk] for kk in range(len(anisbasis)))
        ORF *= 2.0

    # Define frequencies spanning from DC to Nyquist.
    # This is a vector spanning these frequencies in increments of 1/(dur*howml).
    f = np.arange(1 / dur, 1 / (2 * dt), 1 / (dur * howml))
    f[0] = f[1]  # avoid divide by 0 warning
    Nf = len(f)

    # Use Cholesky transform to take 'square root' of ORF
    M = np.linalg.cholesky(ORF)

    # Create random frequency series from zero mean, unit variance, Gaussian distributions
    w = np.zeros((Npulsars, Nf), complex)
    for ll in range(Npulsars):
        w[ll, :] = np.random.randn(Nf) + 1j * np.random.randn(Nf)

    # strain amplitude
    if userSpec is None:

        f1yr = 1 / 3.16e7
        alpha = -0.5 * (gam - 3)
        hcf = Amp * (f / f1yr) ** (alpha)
        if turnover:
            si = alpha - beta
            hcf /= (1 + (f / f0) ** (power * si)) ** (1 / power)

    elif userSpec is not None:

        freqs = userSpec[:, 0]
        if len(userSpec[:, 0]) != len(freqs):
            raise ValueError("Number of supplied spectral points does not match number of frequencies!")
        else:
            fspec_in = interp.interp1d(freqs, userSpec[:, 1], kind="linear")
            fspec_ex = extrap1d(fspec_in)
            hcf = 10.0 ** np.log10(userSpec[:, 1])
            #hcf = 10.0 ** np.log10(fspec_ex(f))
            print("userspec: ", np.log10(userSpec[:, 1]))
            print(hcf)
    plt.plot(hcf)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    plt.clf()

    C = 1 / 96 / np.pi**2 * hcf**2 / f**3 * dur * howml
    
    plt.plot(f, C)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    plt.clf()

    # inject residuals in the frequency domain
    Res_f = np.dot(M, w)
    for ll in range(Npulsars):
        Res_f[ll] = Res_f[ll] * C ** (0.5)  # rescale by frequency dependent factor
        Res_f[ll, 0] = 0  # set DC bin to zero to avoid infinities
        Res_f[ll, -1] = 0  # set Nyquist bin to zero also

    # Now fill in bins after Nyquist (for fft data packing) and take inverse FT
    Res_f2 = np.zeros((Npulsars, 2 * Nf - 2), complex)
    Res_t = np.zeros((Npulsars, 2 * Nf - 2))
    Res_f2[:, 0:Nf] = Res_f[:, 0:Nf]
    Res_f2[:, Nf : (2 * Nf - 2)] = np.conj(Res_f[:, (Nf - 2) : 0 : -1])
    Res_t = np.real(np.fft.ifft(Res_f2) / dt)

    # shorten data and interpolate onto TOAs
    res_gw = []

    for ll in range(Npulsars):
        nt = min(len(ut), Res_t.shape[1])

        ut_use = ut[:nt]
        res_use = Res_t[ll, :nt]

        f = interp.interp1d(
            ut_use,
            res_use,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        res_gw.append(f(psr[ll].toas() * 86400))

    # return res_gw
    ct = 0
    for p in psr:
        p.stoas[:] += res_gw[ct] / 86400.0
        ct += 1

        
def createFreq(
    psr,
    seed=None,
    npts=600,
    cadence=14,
    howml=1
):
    """
    Function to create GW-induced residuals from a stochastic GWB as defined
    in Chamberlin, Creighton, Demorest, et al. (2014).

    :param psr: pulsar object for single pulsar
    :param Amp: Amplitude of red noise in GW units
    :param gam: Red noise power law spectral index
    :param noCorr: Add red noise with no spatial correlations
    :param seed: Random number seed
    :param turnover: Produce spectrum with turnover at frequency f0
    :param clm: coefficients of spherical harmonic decomposition of GW power
    :param lmax: maximum multipole of GW power decomposition
    :param f0: Frequency of spectrum turnover
    :param beta: Spectral index of power spectram for f << f0
    :param power: Fudge factor for flatness of spectrum turnover
    :param userSpec: User-supplied characteristic strain spectrum
                     (first column is freqs, second is spectrum)
    :param npts: Number of points used in interpolation
    :param howml: Lowest frequency is 1/(howml * T)

    :returns: list of residuals for each pulsar
    """

    if seed is not None:
        np.random.seed(seed)

    # number of pulsars
    Npulsars = len(psr)

    # gw start and end times for entire data set
    start = np.min([p.toas().min() * 86400 for p in psr]) - 86400
    stop = np.max([p.toas().max() * 86400 for p in psr]) + 86400

    # duration of the signal
    dur = stop - start

    # get maximum number of points
    if npts is None:
        # default to cadence of 2 weeks
        npts = dur / (86400 * cadence)

    # make a vector of evenly sampled data points
    ut = np.linspace(start, stop, npts)

    # time resolution in days
    dt = dur / npts

    # Define frequencies spanning from DC to Nyquist.
    # This is a vector spanning these frequencies in increments of 1/(dur*howml).
    f = np.arange(1 / dur, 1 / (2 * dt), 1 / (dur * howml))
    f[0] = f[1]  # avoid divide by 0 warning
    Nf = len(f)
    
    return f



def gaussian_kappa_from_sigma_deg(sigma_deg):
    """kappa = 1/sigma^2 for sigma given in degrees."""
    sigma = np.deg2rad(sigma_deg)
    return 1.0 / sigma**2


def gaussian_kappa_from_ellipse_deg(sigma_x_deg, sigma_y_deg, mode="geometric"):
    """Single effective kappa for the isotropic Gaussian ORF.

    The analytic ORF used below is for an axisymmetric Gaussian.  For the
    stringlike injection we use a scalar effective width.  The default is the
    geometric mean, sigma_eff = sqrt(sigma_x*sigma_y), preserving patch area.
    """
    if mode == "geometric":
        sigma_eff_deg = np.sqrt(sigma_x_deg * sigma_y_deg)
    elif mode == "major":
        sigma_eff_deg = sigma_x_deg
    elif mode == "minor":
        sigma_eff_deg = sigma_y_deg
    elif mode == "rms":
        sigma_eff_deg = np.sqrt(0.5 * (sigma_x_deg**2 + sigma_y_deg**2))
    else:
        sigma_eff_deg = float(mode)
    return gaussian_kappa_from_sigma_deg(sigma_eff_deg), sigma_eff_deg


def random_cap_around(ra0, dec0, cap_angle, N):
    """
    Равномерно по телесному углу генерирует N точек внутри конуса
    углового радиуса cap_angle (рад) вокруг (ra0, dec0) (рад).
    """
    c = radec_to_unit(ra0, dec0)

    a = np.array([0.0, 0.0, 1.0]) if abs(c[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, c)
    u /= np.linalg.norm(u)
    v = np.cross(c, u)

    cos_alpha = np.random.uniform(np.cos(cap_angle), 1.0, size=N)
    alpha = np.arccos(cos_alpha)
    beta = np.random.uniform(0.0, 2 * np.pi, size=N)

    dirs = (
        cos_alpha[:, None] * c[None, :]
        + np.sin(alpha)[:, None]
        * (
            np.cos(beta)[:, None] * u[None, :]
            + np.sin(beta)[:, None] * v[None, :]
        )
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    ra, dec = unit_to_radec(dirs)
    return ra, dec, dirs


# координаты
if coord == "cone":
    phi, theta, vecs = random_cap_around(ra0, dec0, cap_angle, Npsr)
else:
    raise ValueError("Этот пример реализует coord='cone'.")

# x,y,z
xi, yi, zi = vecs[:, 0], vecs[:, 1], vecs[:, 2]

# fixed: robust directory creation
os.makedirs(os.path.join(datadir_out, "par"), exist_ok=True)
os.makedirs(os.path.join(datadir_out, "tim"), exist_ok=True)

for numb in range(0, Npsr):
    np.random.seed()

    name_psr = "J0030+0451-simulate"

    psr = T.tempopulsar(
        parfile=datadir_in + name_psr + ".par",
        timfile=datadir_in + name_psr + ".tim"
    )

    # intentionally left unchanged per your request
    pmra, pmdec, px, f0, f1, dm = [
        np.float64(pmra_cat[rand_n[numb]]),
        np.float64(pmdec_cat[rand_n[numb]]),
        px_cat[rand_n[numb]],
        np.float64(f0_cat[rand_n[numb]]),
        np.float64(f1_cat[rand_n[numb]]),
        np.float64(dm_cat[rand_n[numb]])
    ]

    # intentionally left unchanged per your request
    if px != "*":
        psr.vals(
            [phi[numb], theta[numb], pmra, pmdec, np.float64(px), f0, f1, dm],
            which=["RAJ", "DECJ", "PMRA", "PMDEC", "PX", "F0", "F1", "DM"]
        )
        psr["PX"].fit = "True"
        psr["PX"].err = 0.02
    else:
        psr.vals(
            [phi[numb], theta[numb], pmra, pmdec, f0, f1, dm],
            which=["RAJ", "DECJ", "PMRA", "PMDEC", "F0", "F1", "DM"]
        )

    LT.make_ideal(psr)

    psr.name = prefix_psr + str(numb)
    psr.savepar(datadir_out + "par/" + prefix_psr + str(numb) + ".par")

    # fixed: save tim files because they are used later
    psr.savetim(datadir_out + "tim/" + prefix_psr + str(numb) + ".tim")
    T.purgetim(datadir_out + "tim/" + prefix_psr + str(numb) + ".tim")


plt.figure()
for numb in range(Npsr):
    psr = T.tempopulsar(
        parfile=datadir_out + "par/" + prefix_psr + str(numb) + ".par",
        timfile=datadir_out + "tim/" + prefix_psr + str(numb) + ".tim"
    )
    LP.plotres(psr)

# fixed: save instead of show on server
plt.savefig(os.path.join(datadir_out, "initial_residuals.png"), dpi=200, bbox_inches="tight")
plt.close()

outdir = f"{datadir_out}final"                                                                                                  ##############################

# fixed: robust directory creation
os.makedirs(outdir, exist_ok=True)
os.makedirs(os.path.join(outdir, "par"), exist_ok=True)
os.makedirs(os.path.join(outdir, "tim"), exist_ok=True)
os.makedirs(chains_dir, exist_ok=True)                                                                    ##############################

parfiles = sorted(glob.glob(os.path.join(datadir_out, "par", "*.par")))
Npsr = len(parfiles)

psrs = []

for ii in range(0, Npsr):
    psr = LT.fakepulsar(
        parfile=parfiles[ii],
        obstimes=np.arange(53000, 53000 + 10 * 365.25, 28.0),
        toaerr=0.1
    )

    LT.make_ideal(psr)
    LT.add_efac(psr, efac=1.0)
    psrs.append(psr)




gwtheta = np.pi / 2
gwphi = np.pi / 2
Amp = 7e-14
gamma = 0.1
fgw = 3e-8
howml = 2


#make spectrum
#define the spectrum
freq = createFreq(psrs, howml=howml, cadence=28, npts=1000)
print(freq)
index_f = np.argmin(abs(freq-fgw))
spec = 1e-50*np.ones(len(freq))
spec[index_f] = Amp
userSpec = np.asarray([freq, spec]).T

createGWB(
    psrs,
    Amp=Amp,
    gam=gamma,
#    seed=12345,
    gwtheta=gwtheta,
    gwphi=gwphi,
    kappa=KAPPA,
    useGaussianORF=True,
    howml=howml,
    userSpec=userSpec, 
    cadence=28, 
    npts=1000
)


#for ii in range(len(psrs)):
#    LT.add_cstring(
#        psrs[ii],
#        gwtheta,
#        gwphi,
#        h,
#        fgw,
#        phase0,
#        psi,
#        pdist=pdist[ii],
#        psrTerm=True,
#        tref=0
#        )

for Psr in psrs:
    Psr.savepar(outdir + "/par/" + Psr.name + ".par")
    Psr.savetim(outdir + "/tim/" + Psr.name + ".tim")
    T.purgetim(outdir + "/tim/" + Psr.name + ".tim")

# pulsars: RA/Dec
ra_psr = phi
dec_psr = theta

# --- convert to Mollweide coords ---
lon_psr = -(ra_psr - np.pi)
lon_psr = (lon_psr + np.pi) % (2 * np.pi) - np.pi
lat_psr = dec_psr

# source: (gwtheta, gwphi) -> RA/Dec
ra_src = gwphi
dec_src = np.pi / 2 - gwtheta
lon_src = -(ra_src - np.pi)
lon_src = (lon_src + np.pi) % (2 * np.pi) - np.pi
lat_src = dec_src

# --- plot ---
plt.figure(figsize=(9, 5))
ax = plt.subplot(111, projection="mollweide")
ax.set_title("Mollweide: pulsars + GW source")
ax.grid(True)

ax.plot(lon_psr, lat_psr, "o", ms=5, alpha=0.8, label=f"Pulsars (N={len(lon_psr)})")
ax.plot(lon_src, lat_src, "*", ms=14, label="GW source")
#if "src_ra" in globals():
#    lon_src_cloud = -(src_ra - np.pi)
#    lon_src_cloud = (lon_src_cloud + np.pi) % (2 * np.pi) - np.pi
#    ax.plot(lon_src_cloud, src_dec, ".", ms=2, alpha=0.35, label=f"String sources (N={len(src_ra)})")

ax.legend(loc="lower left")
plt.savefig(os.path.join(outdir, "mollweide.png"), dpi=200, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(8, 4))

for p in psrs:
    t = p.toas()
    res_us = p.residuals() * 1e6
    ax.plot(t, res_us, ".", markersize=2, alpha=0.5)

ax.set_xlabel("TOA, MJD")
ax.set_ylabel(r"$\delta t$, $\mu$s")
ax.grid(True)

ax2 = ax.twinx()

for p in psrs:
    t = p.toas()
    res_phase = p.residuals() * p["F0"].val
    ax2.plot(t, res_phase, "^", markersize=2, alpha=0.5)

ax2.set_ylabel(r"$\delta t / P$")

plt.tight_layout()
plt.savefig(os.path.join(outdir, "residuals.png"), dpi=200, bbox_inches="tight")
plt.close()

parfiles = sorted(glob.glob(os.path.join(outdir, "par", "*.par")))
timfiles = sorted(glob.glob(os.path.join(outdir, "tim", "*.tim")))

psrs = []
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t)
    psrs.append(psr)

# find the maximum time span to set GW frequency sampling
tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
Tspan = np.max(tmax) - np.min(tmin)









##### parameters and priors #####

efac = parameter.Constant(1.0)

log10_A = parameter.Uniform(-18, -13)
gamma = parameter.Constant(0.1)

##### Set up signals #####

ef = white_signals.MeasurementNoise(efac=efac)

pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

gwtheta =  parameter.Constant(np.pi/2.)("gw_theta")
gwphi = parameter.Constant(np.pi/2.)("gw_phi")
kappa = parameter.Uniform(0.1, 300)("gw_kappa")














@function
def orf_gaussian(pos1, pos2, gwtheta, gwphi, kappa):
    Omega0 = _gw_source_unit_vector(gwtheta, gwphi)
    val = K_exp(Omega0, pos1, pos2, kappa)
    if np.allclose(pos1, pos2):
        val += 1e-6
    return val


orf = orf_gaussian(gwtheta=gwtheta, gwphi=gwphi, kappa=kappa)

flis = np.linspace(fgw, fgw+0.5/Tspan, 1)
print(fgw)
print(flis)
crn = gp_signals.FourierBasisCommonGP(pl, orf, fmin=fgw, fmax=fgw+0.5/Tspan, components=1, name="gw", Tspan=Tspan)

tm = gp_signals.TimingModel()

model = ef + tm + crn

pta = signal_base.PTA([model(psr) for psr in psrs])

xs = {par.name: par.sample() for par in pta.params}
ndim = len(xs)

cov = np.diag(np.ones(ndim) * 0.1**2)

ndim = len(xs)
groups = [range(0, ndim)]
groups.extend(map(list, zip(range(0, ndim, 2), range(1, ndim, 2))))


list_kappa=np.arange(0.1, 70, 1)

list_likel = [pta.get_lnlikelihood([list_kappa[j], 2e-14]) for j in range(len(list_kappa))]

print(list_kappa, list_likel)

plt.plot(list_kappa, list_likel)
plt.axvline(KAPPA, color="grey")
plt.savefig("fig.png", dpi=300)

def ln_likely(x, **kwargs):
    if np.isfinite(pta.get_lnlikelihood(x)):
        return pta.get_lnlikelihood(x)
    else:		
        return -np.inf

sampler = ptmcmc(
    ndim,
    ln_likely,
    pta.get_lnprior,
    cov,
    groups=groups,
    outDir = chains_dir + "/"                                                                                       ##############################
)

N = 100000
x0 = np.hstack([p.sample() for p in pta.params])
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)

chain = np.loadtxt(f"{chains_dir}/chain_1.txt")                                                                    ##############################
burn = int(0.25 * chain.shape[0])

fig1 = corner.corner(
    chain[burn:, :ndim][:, [pta.param_names.index("gw_phi"), pta.param_names.index("gw_theta")]],
    40,
    labels=["gw_phi", "gw_theta"],
    smooth=True,
    truths=[np.pi / 2, np.pi / 2]
)
fig1.savefig(os.path.join(outdir, "corner_gw_position.png"), dpi=200, bbox_inches="tight")
plt.close(fig1)

plt.hist(chain[burn:, :ndim][:, pta.param_names.index("gw_log10_A")])
plt.savefig(os.path.join(outdir, "corner_gw_spectrum.png"), dpi=200, bbox_inches="tight")

plt.hist(chain[burn:, :ndim][:, pta.param_names.index("gw_kappa")])
plt.savefig(os.path.join(outdir, "corner_gw_kappa.png"), dpi=200, bbox_inches="tight")

print("Done.", flush=True)
print(f"Input dir: {datadir_in}", flush=True)
print(f"Intermediate dir: {datadir_out}", flush=True)
print(f"Final dir: {outdir}", flush=True)
print("Saved plots:", flush=True)
print(f"  {os.path.join(datadir_out, 'initial_residuals.png')}", flush=True)
print(f"  {os.path.join(outdir, 'mollweide.png')}", flush=True)
print(f"  {os.path.join(outdir, 'residuals.png')}", flush=True)
print(f"  {os.path.join(outdir, 'corner_gw_position.png')}", flush=True)
print(f"  {os.path.join(outdir, 'corner_gw_spectrum.png')}", flush=True)

