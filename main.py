import pta_rec as pta
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

Npsr = 50
map_nside = 1000

psr_seed = 42
gwb_seed = 42
map_seed = 42

cadence = 28.0
period = 10.0
toaerr = 1e-1

Amp = 7e-13#1e-11
fgw = 3e-8
gamma = 0.1

Omega_0 = pta.to_unit(0, 0)
sigma = 30
kappa = 1/(np.deg2rad(sigma)**2)
alpha = 3
beta = 2
ell_break = 10


P_Omega = pta.generate_power_map(
    nside=map_nside,
    alpha=alpha, 
    map_seed=map_seed
)

# P_Omega = pta.generate_phisical_map(
#     nside=map_nside,
#     beta=beta, 
#     ell_break=ell_break,
#     map_seed=map_seed
# )


# P_Omega = pta.spherical_harmonic(
#     nside=map_nside,
#     l=10, m=0
# )


# P_Omega = pta.gaussian_spec(
#     Omega=pta.get_Omega(map_nside), 
#     Omega_0=Omega_0, 
#     kappa=kappa
# )

f_spec = 'pow'
P_spec = 'numeric'

f_params = {
    'pow' : {
        'A':Amp, 
        'gamma': gamma,
    },
    'delta' : {
        'A':Amp, 
        'f_0': fgw,
    }
} 

P_params = {
    'iso': {},
    'point': {
        'Omega_0': Omega_0
    },
    'gauss': {
        'Omega_0': Omega_0,
        'kappa': kappa
    },
    'numeric': {
        'P_spec': P_Omega,
    }
}

iom = pta.IOManager(console_log='debug')
gw = pta.GravitationalWaves()
pc = pta.PulsarCatalog(iom)
po = pta.PulsarObservation(iom, gw)

iom.setup(exp_name="test")
pc.setup(
    catalog='pta'
)
po.setup(
    model='J0030+0451', 
    period=period, 
    cadence=cadence, 
    toaerr=toaerr, 
    start=53000, 
    efac=1.0
)
gw.setup(
    f_spec=f_spec, 
    P_spec=P_spec, 
    f_params=f_params[f_spec], 
    P_params=P_params[P_spec], 
    cadence=cadence,
    gwb_seed=gwb_seed,
)


def make_analysis(orf, key = None):
    Gamma, Xi = [], []
    for i in range(len(pulsars)):
        for j in range(len(pulsars)):
            if i == j:
                continue
            p1, p2 = pta.to_vec(pulsars[i]), pta.to_vec(pulsars[j])
            xi = pta.angular_distance(p1, p2)
            gamma = orf[i, j]
            Xi.append(xi)
            Gamma.append(gamma)

    Xi, Gamma = np.array(Xi), np.array(Gamma)
    Xi_mean, Gamma_mean, Gamma_std = pta.average(Xi, Gamma)

    Theta = np.linspace(0, pta.PI, 1000)
    Mu = []
    for k in range(1000):
        Mu.append(pta.hellings_downs_curve(Theta[k]))

    cors = Theta, Mu, Xi, Gamma
    stat = Theta, Mu, Xi_mean, Gamma_mean, Gamma_std

    pta.plot_ORF(cors, iom=iom, key=key, show=False)
    pta.plot_HD_stat(stat, iom=iom, key=key, show=False)

Omega, P = gw.make_P_spec(nside=map_nside, params=P_params[P_spec])
pta.plot_map(iom, data=P, log=False)
pta.plot_Cl(iom, P, lmax=100)

catalog = pc.sample_catalog(n=Npsr, seed=psr_seed, tempo=True)
pulsars = po.make_gwb_observations(catalog, load_mod=True, load_obs=True, save=False)
HD = pta.eval_ORF(pulsars)
ORF = gw.orf
print(np.diag(HD))
print(np.diag(ORF))
make_analysis(HD, key='eval')
make_analysis(ORF, key='true')

# catalog = pc.load_catalog(name='real', path="test-obs")
# pulsars = po.load_observations(catalog, path="test-obs")

# catalog = pc.load_catalog(name=f'sample_{psr_seed}')
# pulsars = po.load_observations(catalog, gwb=True)


