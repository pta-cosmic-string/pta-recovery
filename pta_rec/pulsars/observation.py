import numpy as np
import libstempo as T
import libstempo.toasim as LT

class PulsarObservation:
    def __init__(self, iom, gw):
        self.iom = iom
        self.gw = gw

    def setup(self, model="J0000+0000", period=10, cadence=28.0, start=53000, toaerr=0.1, efac=1.0):
        self.iom.check_model_psr(model)
        self.model = model 
        self.psr_model = str(self.iom.storage_path / f"{model}")
        self.mod_path = str(self.iom.mod_path.relative_to(self.iom.path))
        self.obs_path = str(self.iom.obs_path.relative_to(self.iom.path))
        self.gwb_path = str(self.iom.gwb_path.relative_to(self.iom.path))
        
        self.obs_time = np.arange(start, start + period * 365.25, cadence)
        self.toaerr = toaerr
        self.efac = efac

    def load_model(self, path=None):
        if path is None:
            path = self.mod_path
        psr = T.tempopulsar(parfile=f"{path}.par", timfile=f"{path}.tim")
        return psr

    def load_observation(self, params, path=None):
        if path is None:
            path = self.obs_path
        psr = self.load_model(path=f"{path}/{params['PSRJ']}")
        return psr

    def load_observations(self, catalog, gwb=False, path=None):
        if path is None:
            if gwb:
                path = self.gwb_path
                obs_type = "GWB"
            else:
                path = self.obs_path
                obs_type = "Clear"
        else:
            obs_type = "Real"

        pulsars = []
        for k in range(len(catalog)):
            params = catalog.iloc[k]
            psr = self.load_observation(params, path)
            pulsars.append(psr)

        self.iom.info(f"Load Observed {obs_type} TOAs for {catalog.name} catalog")
        return pulsars

    def save_model(self, psr, path=None, tims=False):
        if path is None:
            path = self.mod_path
        psr.savepar(f"{path}/{psr.name}.par")
        if tims:
            psr.savetim(f"{path}/{psr.name}.tim")
            T.purgetim(f"{path}/{psr.name}.tim")

    def save_observaton(self, psr, path=None):
        if path is None:
            path = self.obs_path
        self.save_model(psr, path, tims=True)

    def make_model(self, params):
        psr = self.load_model(self.psr_model)
        keys = ["RAJ", "DECJ", "PMRA", "PMDEC", "F0", "F1", "DM"]
        psr.vals(params[keys], which=keys)
        if not np.isnan(params['PX']):
            psr.vals([params['PX']], which=['PX'])
            psr["PX"].fit = "True"
            psr["PX"].err = 0.02
        psr.name = params['PSRJ']
        LT.make_ideal(psr)
        self.save_model(psr, f"{self.mod_path}")

    def make_models(self, catalog):
        for k in range(len(catalog)):
            params = catalog.iloc[k]
            self.make_model(params)
        self.iom.info(f"Make Ideal TOAs for {catalog.name} catalog")

    def make_observation(self, params, save=True): 
        name = params['PSRJ']
        psr = LT.fakepulsar(
            parfile=f"{self.mod_path}/{name}.par", 
            obstimes=self.obs_time, 
            toaerr=self.toaerr
        )
        LT.make_ideal(psr)
        LT.add_efac(psr, efac=self.efac)
        self.save_observaton(psr)
        return psr

    def make_observations(self, catalog, save=True):
        pulsars = []
        for k in range(len(catalog)):
            params = catalog.iloc[k]
            psr = self.make_observation(params, save)
            pulsars.append(psr)
        self.iom.info(f"Make Observed Clear TOAs for {catalog.name} catalog")
        return pulsars

    def make_gwb_observations(self, catalog, load_mod=False, load_obs=False, save=True):
        if not load_mod:
            self.make_models(catalog)
        if not load_obs:
            pulsars = self.make_observations(catalog)
        else:  
            pulsars = self.load_observations(catalog)

        pulsars = self.gw.make_gwb(pulsars)
        if save:
            for p in pulsars:
                self.save_observaton(p, path=self.gwb_path)
        self.iom.info(f"Make Observed GWB TOAs for {catalog.name} catalog")
        return pulsars

