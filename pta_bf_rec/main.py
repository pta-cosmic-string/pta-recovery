import healpy as hp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class PulsarArray:
    def __init__(self, n_array, radius=1, seed_psr=42):
        self.n_array = n_array
        self.radius = radius
        self.rng_psr = np.random.default_rng(seed=seed_psr)
        self._init_array()

    def _init_array(self):
        phi = self.rng_psr.uniform(0, 2*np.pi, self.n_array)
        cos_theta = self.rng_psr.uniform(-1, 1, self.n_array)
        theta = np.arccos(cos_theta)

        rho = self.radius * self.rng_psr.uniform(0, 1, self.n_array) ** (1/3)

        x = rho * np.sin(theta) * np.cos(phi)
        y = rho * np.sin(theta) * np.sin(phi)
        z = rho * np.cos(theta)

        self.pulsar_arr = np.array([x, y, z]).T
        self.pulsar_dist = rho.T
        self.pulsar_vec = self.pulsar_arr / np.tile(self.pulsar_dist[:,None], (1, 3))
        self.n_pulsars = self.pulsar_vec.shape[0]

    def get_pixs(self, skymap):
        vec = self.pulsar_vec
        pix_idx = hp.vec2pix(skymap.nside, vec[:,0], vec[:,1], vec[:,2], nest=False)
        data = np.zeros((skymap.npix))
        data[pix_idx] = 1.0
        return data
    
    def plot(self, show=True, save=True):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.pulsar_arr.T[0], self.pulsar_arr.T[1], self.pulsar_arr.T[2], c='blue', s=20, label='PTA')
        ax.scatter(0, 0, 0, c='red', s=100, label='Observer')

        ax.set_box_aspect([1, 1, 1])

        ax.set_xlabel("x, kpc")
        ax.set_ylabel("y, kpc")
        ax.set_zlabel("z, kpc")
        ax.set_title("Distribution of Pulsar Array")
        ax.legend()
        
        if not os.path.exists('data'):
            os.mkdir('data')
        if save:
            plt.savefig('data/PTA.png')
        if show:
            plt.show()

class SkyMap:
    def __init__(self, nside = 8):
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)

        self._init_angles()
        self._init_vectors()

    def _init_angles(self):        
        self.theta, self.phi = hp.pix2ang(self.nside, np.arange(self.npix))
        self.dOmega = hp.nside2pixarea(self.nside)
    
    def _init_vectors(self):
        self.vectors = np.array(hp.pix2vec(self.nside, np.arange(self.npix)))

        self.Omega = np.array([np.sin(self.theta)*np.cos(self.phi), 
                               np.sin(self.theta)*np.sin(self.phi), 
                               np.cos(self.theta)])
        
        self.m = np.array([+np.sin(self.phi), 
                           -np.cos(self.phi),
                           +np.zeros(self.phi.shape)])
        self.n = np.array([-np.cos(self.theta)*np.cos(self.phi), 
                           -np.cos(self.theta)*np.sin(self.phi), 
                           +np.sin(self.theta)])
                
        self.e_p = np.einsum('ik,jk->ijk', self.m, self.m) - np.einsum('ik,jk->ijk', self.n, self.n) 
        self.e_c = np.einsum('ik,jk->ijk', self.m, self.n) + np.einsum('ik,jk->ijk', self.n, self.m) 
        
        self.e = self.e_p + 1j * self.e_c

    def get_data(self, func, **kwargs):
        return func(self.theta, self.phi, **kwargs)

    def plot(self, data=None, data_vec=None, title="Map", unit="$P(\Omega)$", cmap='viridis', cbar=True, show=True, save=True, filename='map'):
        if data is None:
            data = np.arange(self.npix)
        
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
    
        if not os.path.exists('data'):
            os.mkdir('data')
        if save:
            plt.savefig(f'data/{filename}.png', dpi=1000, bbox_inches='tight')
        if show:
            plt.show()

def circle_on_sphere(center_theta, center_phi, radius_rad, n_points=100):
    # Center in Cartesian
    x0 = np.sin(center_theta) * np.cos(center_phi)
    y0 = np.sin(center_theta) * np.sin(center_phi)
    z0 = np.cos(center_theta)
    c = np.array([x0, y0, z0])

    # Build orthonormal basis in the tangent plane at center
    # Choose arbitrary vector not parallel to c (here (0,0,1) unless c is near z-axis)
    if abs(z0) < 0.99999:
        u_vec = np.cross(c, [0, 0, 1])
    else:
        u_vec = np.cross(c, [1, 0, 0])
    u_vec = np.cross(c, [0, 0, 1])
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(c, u_vec)
    v_vec /= np.linalg.norm(v_vec)

    # Parametric angles around the circle
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Precompute sin and cos of radius
    cos_r = np.cos(radius_rad)
    sin_r = np.sin(radius_rad)

    # Points on circle: p = c * cos(r) + (u*cos(alpha) + v*sin(alpha)) * sin(r)
    points = []
    for alpha in angles:
        p = cos_r * c + sin_r * (np.cos(alpha) * u_vec + np.sin(alpha) * v_vec)
        points.append(p)
    points = np.array(points)
    return points


def gaussian(Omega, Omega_0, kappa):
    delta_Omega = np.einsum('i...,ji->j...', Omega, Omega_0)
    gauss  = 2*kappa/(1 - np.exp(-2*kappa)) * np.exp(kappa * (delta_Omega-1))
    return gauss

def K_pq(Omega, p1, p2):
    a = np.einsum('i...,i->...', Omega, p1)
    b = np.einsum('i...,i->...', Omega, p2)
    c = np.dot(p1, p2)
    
    eps = 1e-16
    denominator = (1 + a) * (1 + b)
    denominator = np.where(np.abs(denominator) < eps, eps, denominator)
    
    term1 = 2 * (c - a * b)**2 / denominator
    term2 = (1 - a) * (1 - b)
    return 0.75 * (term1 - term2)

def _scaled_expi(w, switch=100.0, max_terms=50):
    """
    Вычисляет exp(-w) * Ei(w) устойчиво.
    Для |w| < switch — напрямую.
    Для |w| >= switch — через асимптотический ряд.
    """
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
    norm = 2 * k * np.exp(-k * (a + 1.0))/(-np.expm1(-2.0 * k))
    return norm * np.real(_scaled_expi(w))

def K_exp(Omega, p1, p2, kappa):
    a = np.einsum('...i,i->...', Omega, p1)
    b = np.einsum('...i,i->...', Omega, p2)
    c = np.dot(p1, p2)
    V = np.sqrt(1 + 2*a*b*c - c**2 - a**2 - b**2)
    k = kappa
    t = (a + b)/(1 + c)
    s = V/(1 + c) 

    L_g = (
        + expi_stable(t, s, a, k)
        + expi_stable(t, s, b, k)
        - expi_stable(t, s, -1, k)
        - expi_stable(t, s, +1, k)
    )
        
    D_g = (
        + (b - c*a)/(1 - a**2) * (1/np.tanh(k) - a * (np.exp(-k*(1+a))/a * 2/(1 - np.exp(-2*k)) + 1))
        + (a - c*b)/(1 - b**2) * (1/np.tanh(k) - b * (np.exp(-k*(1+b))/b * 2/(1 - np.exp(-2*k)) + 1))
        + (c - 3*a*b)/6 * (1/np.tanh(k)*3/k - 3/k**2 - 1)
        - (a+b)/2 * (1/np.tanh(k) - 1/k)
    )

    return 3 * (
        + 1/3 - 1/6 * (1 - c)/2
        + (1 - c)/2 * L_g
        + 1/2 * D_g
    )

def Gamma(SM, PA, P_spec):
    P = PA.pulsar_vec 
    G = np.zeros((len(P), len(P)))
    Dist = np.zeros((len(P), len(P)))
    for i in range(len(P)):
        for j in range(len(P)):
            if i!=j:
                G[i, j] = 1/(4 * np.pi) * np.sum(K_pq(SM.Omega, P[i], P[j]) * P_spec * SM.dOmega)
                Dist[i, j] = np.arccos(np.dot(P[i], P[j]))
    return G, Dist


def plot_pulsars(PA,SM):
    SM.plot(data=np.zeros(SM.npix), data_vec=PA.pulsar_vec, title='Pulsar map', cbar=False)

def plot_spectrum(PA,SM):
    Omega_0 = hp.ang2vec(0, 0, lonlat=True)
    kappa = 1
    P_exp = gaussian(SM.Omega, Omega_0, kappa)
    SM.plot(data=P_exp, data_vec=np.array([Omega_0]))

def plot_response(PA, SM):
    xi = 60
    p1 = hp.ang2vec(90+xi/2, 0, lonlat=True)
    p2 = hp.ang2vec(90-xi/2, 0, lonlat=True)
    p3 = hp.ang2vec(90, 0+xi/2, lonlat=True)
    p4 = hp.ang2vec(90, 0-xi/2, lonlat=True)
    R = 0
    R += K_pq(SM.Omega, p1, p3)
    R += K_pq(SM.Omega, p1, p4)
    R += K_pq(SM.Omega, p2, p3)
    R += K_pq(SM.Omega, p2, p4)
    SM.plot(data=R**2, data_vec=np.array([p1,p2,p3,p4]))



class ExpBasis:
    def __init__(self, nside=1):
        self.map = SkyMap(nside=nside)
        self.npix = self.map.npix
        self.Omega = self.map.Omega.T
        self.A = np.ones(self.npix)
        self.sigma = np.sqrt(self.map.dOmega/np.pi)
        self.kappa = 1/self.sigma**2

    def get_spec(self, SM):
        G = gaussian(SM.Omega, self.Omega, self.kappa)
        return np.einsum('j...,j->...', G, self.A)

    def get_ORF(self, SM, PA):
        P = PA.pulsar_vec
        ORF = np.zeros((len(P), len(P), self.npix))
        Dist = np.zeros((len(P), len(P), self.npix))
        for i in range(len(P)):
            for j in range(len(P)):
                if i!=j:
                    ORF[i, j, :] = K_exp(self.Omega, P[i], P[j], self.kappa)
                    Dist[i, j, :] = np.arccos(np.dot(P[i], P[j]))
        return ORF, Dist

    def get_grid(self):
        return self.A

    def learn(self, DATA, MODEL):
        d = DATA
        R = MODEL
        A_star = np.linalg.inv(R.T @ R) @ R.T @ d
        self.A = A_star

    def plot_boards(self):
        points = []
        for k in range(self.map.npix):
            circ = circle_on_sphere(self.map.theta[k], self.map.phi[k], radius_rad=1/np.sqrt(self.kappa))
            for p in circ:
                points.append(p)
        self.map.plot(data_vec=np.array(points), title='Exp Basis', cbar=False)

def generate_map(SM, alpha=4):
    # ========================
    # Настройки генерации карты
    # ========================
    nside = SM.nside          # разрешение карты (число пикселей ~12 * nside^2)
    lmax = nside - 1 # максимальный мультиполь (обычно 3*nside-1)
    alpha = alpha       # показатель степенного закона: C_l ∝ l^{-alpha}

    # Создаём массив мультиполей ℓ от 0 до lmax
    ell = np.arange(lmax + 1, dtype=float)

    # Строим спектр мощности C_ℓ
    Cl = np.zeros_like(ell)
    # Начинаем с ℓ = 2, чтобы избежать расходимости в ℓ = 0,1
    mask = ell >= 1
    Cl[mask] = ell[mask] ** (-alpha)   # C_ℓ = ℓ^{-α}

    # (Опционально) Установка seed для воспроизводимости результатов
    np.random.seed(1)

    # Генерация случайной карты на сфере с заданным спектром
    # Функция synfast использует заданные C_ℓ для создания синтетической карты
    map_data = hp.synfast(Cl, nside, lmax=lmax)

    p_data = np.abs(map_data)**2
    norm = np.sum(p_data * SM.dOmega) / (4 * np.pi)
    p_data_norm = p_data/norm

    return p_data_norm

if __name__ == '__main__':
    n_pulsars = 200
    PA = PulsarArray(n_pulsars) 
    SM = SkyMap(nside=100)
    
    EB = ExpBasis(nside=5)
    # EB.plot_boards()
    # SM.plot(data=EB.get_spec(SM), data_vec=EB.Omega)
    # # EB.map.plot(data=EB.get_grid())
    # ORF, DIST = EB.get_ORF(SM, PA)
    i, j = np.arange(n_pulsars), np.arange(n_pulsars)
    I, J = np.meshgrid(i,j, indexing='ij')
    p_i, p_j = PA.pulsar_vec, PA.pulsar_vec
    I, J = np.meshgrid(i,j, indexing='ij')
    
    # GORF = ORF[I<J, :]
    # GDIST = DIST[I<J, :]

    P_data = generate_map(SM, alpha=2)
    ORF_data, DIST_data = Gamma(SM, PA, P_data)
    ORF_mod, DIST_mod = EB.get_ORF(SM, PA)
    # SM.plot(data=P_data, data_vec=EB.Omega)

    DATA = ORF_data[I<J]
    MODEL = ORF_mod[I<J, :]
    print(DATA.shape)
    print(MODEL.shape)
    EB.learn(DATA, MODEL)
    P_model = EB.get_spec(SM)
    SM.plot(data=P_data, filename='data', title='Data')
    SM.plot(data=P_model, filename='clean-map', title='Clean Map')
    SM.plot(data=P_model-P_data, filename='dirty-map', title='Dirty Map')
   

    # DIST_ = DIST_data[I<J, :]

    # idx = 0
    # plt.scatter(DIST_data[I<J], DATA)
    # plt.grid()
    # plt.show()


    
    # xi = 60
    # p1 = hp.ang2vec(90+xi/2, 0, lonlat=True)
    # p2 = hp.ang2vec(90-xi/2, 0, lonlat=True)

    # Omega_0 = hp.ang2vec(-90, 0, lonlat=True)
    # kappa = 1
    # P_exp = gaussian(SM.Omega, Omega_0, kappa)

    # G = Gamma(PA, SM, p1, p2, P_exp)
    # G1 = K_exp(Omega_0, p1, p2, kappa)
    # print(np.abs((G-G1)/G1) * 100)