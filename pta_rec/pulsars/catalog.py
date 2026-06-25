import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PulsarCatalog:
    def __init__(self, iom):
        self.iom = iom
        self.pta_cat = self.iom.open_pta_catalog()
        self.atnf_cat = self.iom.open_atnf_catalog()
        self.atnf = self.iom.open_atnf()

    def setup(self, catalog='pta'):
        if catalog == 'pta':
            self.catalog = self.pta_cat
            self.catalog.name = catalog
        elif catalog == 'atnf':
            self.catalog = self.atnf_cat
            self.catalog.name = catalog
        elif catalog == 'full-atnf':
            self.catalog = self.atnf
            self.catalog.name = catalog
        else:
            self.iom.error(f"There is no '{catalog}' pulsar catalog")

    def tempo_format(self, catalog):
        tempo_cat = catalog.rename(columns={'RAJD': 'RAJ', 'DECJD': 'DECJ'})
        tempo_cat['RAJ'] = np.deg2rad(tempo_cat['RAJ'])
        tempo_cat['DECJ'] = np.deg2rad(tempo_cat['DECJ'])
        tempo_cat.name = catalog.name
        return tempo_cat

    def sample_catalog(self, n, seed=42, name="sample", set_catalog=True, save=True, tempo=False):
        sample = self.catalog.sample(n, random_state=seed)
        sample.name = f"{name}_{seed}"
        self.iom.info(f"Make {sample.name} Pulsar Catalog")
        if set_catalog:
            self.catalog = sample
        if save:
            self.save_catalog(sample)
        if tempo:
            sample = self.tempo_format(sample)
        return sample

    def save_catalog(self, catalog=None):
        if catalog is None:
            catalog = self.catalog
        self.iom.save_catalog(catalog)

    def load_catalog(self, name, path=None, set_catalog=True, tempo=False):
        catalog = self.iom.load_catalog(name, path)
        catalog.name = name
        if set_catalog:
            self.catalog = catalog
        if tempo:
            catalog = self.tempo_format(catalog)
        return catalog

    def compare_catalogs(self, show=False):
        names, pulsars = [], []
        for k in range(len(self.pta_cat)):
            compare = pd.concat([self.pta_cat.iloc[[k]], self.atnf_cat.iloc[[k]]], ignore_index=True)
            compare.insert(0, 'catalog', ['pta', 'atnf'])
            compare.drop(columns=['PSRJ'], inplace=True)
            names.append(self.pta_cat.iloc[k]['PSRJ'])
            pulsars.append(compare)

        if show:
            for k in range(len(pulsars)):
                print(f"{k:<2} | {names[k]}")
                print(pulsars[k])
                print("\n" + "-"*50)

        return names, pulsars

    def plot_catalog(self, catalog=None, show=False, save=True):
        if catalog is None:
            catalog = self.catalog
        rajd_cat, decjd_cat = catalog['RAJD'], catalog['DECJD']
        rajd_cat = np.where(rajd_cat>=180, rajd_cat-360, rajd_cat)
        rajd_cat *= np.pi/180
        decjd_cat *= np.pi/180
        plt.figure(figsize=(8,4))
        plt.subplot(projection="mollweide")
        plt.title(f"{catalog.name} Pulsar Catalogue")
        plt.grid(True)
        plt.plot(rajd_cat, decjd_cat, 'o', markersize=2)
        if save:
            plt.savefig(self.iom.img_path / "psr_cat.png", dpi=1000)
        if show:
            plt.show()
        plt.close()