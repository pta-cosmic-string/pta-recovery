import copy
import libstempo as T

from .funcs import *
from .f_spec import *
from .P_spec import *
from .noise import *

class GravitationalWaves:
    def __init__(self):
        pass

    def setup(self, f_spec='pow', P_spec='iso', f_params={}, P_params={}, cadence=28, gwb_seed=42):
        self.f_spec = f_specs[f_spec]
        self.P_spec = P_specs[P_spec]['P_spec']
        self.hellings_downs = P_specs[P_spec]['hd']
        self.f_params = f_params
        self.P_params = P_params
        self.cadence = cadence
        self.gwb_seed = gwb_seed

    def make_f_spec(self, pulsars, params={}, cadence=28):
        self.Tspan = get_tspan(pulsars)
        self.f_nyq = 1/2 * 365/cadence * F_YEAR
        self.nfreq = np.floor(self.f_nyq * self.Tspan).astype(int)
        self.f_max = self.nfreq / self.Tspan
        self.freqs = make_freqs(self.Tspan, nfreq=self.nfreq)
        self.df = make_dfreq(self.freqs)
        self.h_c = self.f_spec(self.freqs, **params)
        self.psd = residual_psd(self.freqs, self.h_c)
        return self.freqs, self.psd

    def make_P_spec(self, nside=5, params={}):
        self.Omega = get_Omega(nside)
        self.P = self.P_spec(self.Omega, **params)
        return self.Omega, self.P

    def make_orf(self, pulsars, params={}):
        orf = calc_ORF(pulsars, self.hellings_downs, params) 
        return orf

    def make_gwb(self, pulsars):
        self.orf = self.make_orf(pulsars, params=self.P_params)
        freqs, psd = self.make_f_spec(pulsars, params=self.f_params, cadence=self.cadence)
        df = make_dfreq(freqs)
        a, b = gwb_coefficients(self.orf, freqs, df, psd, self.gwb_seed)

        for ip, psr in enumerate(pulsars): 
            toas = get_toas(psr) * DAY
            F = fourier_design_matrix(toas,freqs)
            coeff = np.empty(2 * len(freqs))
            coeff[0::2] = a[ip]
            coeff[1::2] = b[ip]
            residuals = F @ coeff
            psr.stoas[:] += residuals / DAY

        return pulsars