import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import libstempo.plot as LP
import libstempo as T

def plot_HD_stat(data, iom, key=None, show=False, save=True):
    Theta, Mu, Xi_mean, Gamma_mean, Gamma_std = data
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(np.rad2deg(Theta), Mu, '--', color='violet', linewidth=2, label='Predicted (HD)')
    ax.errorbar(np.rad2deg(Xi_mean), Gamma_mean, yerr=Gamma_std, fmt='o', color='blue', 
                capsize=4, ecolor='blue', elinewidth=1.5, markersize=8, label='Statistics')
    ax.set_xlabel('Angular distance between pulsars, $\\xi_{ab}$ [deg]', fontsize=12)
    ax.set_ylabel('Hellings Downs, $\\Gamma_{ab}(\\xi_{ab})$', fontsize=12)
    ax.grid(True, alpha=0.6)
    ax.legend()

    prefix = ""
    if key is not None:
        prefix = f"_{key}"
    if save:
        plt.savefig(f"{iom.img_path}/HD{prefix}.png", dpi=1000)
    if show:
        plt.show()

    plt.close()

def plot_ORF(data, iom, key=None, show=False, save=True):
    Theta, Mu, Xi, Gamma = data
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(np.rad2deg(Xi), Gamma, color='blue', alpha=0.05)
    ax.plot(np.rad2deg(Theta), Mu, '--', color='black', linewidth=2, label='Predicted (HD)')
    ax.set_xlabel('Angular distance between pulsars, $\\xi_{ab}$ [deg]', fontsize=12)
    ax.set_ylabel('Hellings Downs, $\\Gamma_{ab}(\\xi_{ab})$', fontsize=12)
    ax.grid(True, alpha=0.6)
    ax.legend()

    prefix = ""
    if key is not None:
        prefix = f"_{key}"
    if save:
        plt.savefig(f"{iom.img_path}/orf{prefix}.png", dpi=1000)
    if show:
        plt.show()
    plt.close()


def plot_catalog(catalog, iom, show=False, save=True):
    plt.figure()
    for k in range(len(catalog)):
        name = catalog['PSRJ'].iloc[k]
        psr = T.tempopulsar(
            parfile=f"{iom.obs_path}/{name}.par",
            timfile=f"{iom.obs_path}/{name}.tim",
        )
        LP.plotres(psr)
    if save:
        plt.savefig(iom.img_path / "residuals.png", dpi=1000)
    if show:
        plt.show()
    plt.close()

def plot_map(iom, data=None, data_vec=None, log=False, title="Map", unit="$P(\Omega)$", cmap='viridis', cbar=True, show=False, save=True):
    if data is not None:
        if log:
            data = np.log10(1+data)
        m = hp.mollview(
        data, 
        title=title,
        unit=unit, 
        cmap=cmap,
        cbar=cbar,
        )
    
    hp.graticule()
    
    if data_vec is not None:
        lat, lon = hp.pixelfunc.vec2ang(data_vec, lonlat=True) 
        hp.projplot(lat, lon, '.', color='red', markersize=10, lonlat=True)
    
    if save:
        plt.savefig(iom.img_path / "map.png", dpi=1000)
    if show:
        plt.show()
    plt.close()

def plot_Cl(iom, data_map, l_start=2, lmax=100, show=False, save=True):
    cl = hp.anafast(data_map, lmax=lmax)
    ell = np.arange(len(cl))

    plt.plot(ell[l_start::], cl[l_start::])
    plt.xlabel('Multipole l')
    plt.ylabel('$C_l$')
    plt.loglog()
    
    if save:
        plt.savefig(iom.img_path / "Cl.png", dpi=1000)
    if show:
        plt.show()
    plt.close()