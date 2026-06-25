import os, shutil, pathlib
import logging, colorlog
import pandas as pd
import numpy as np
import psrqpy

from urllib.request import urlretrieve

model_psr_source = 'https://raw.githubusercontent.com/nanograv/enterprise/master/tests/data/mdc1'

standart_datefmt = '%Y-%m-%d %H:%M:%S'
short_datefmt = '%y-%m-%d %H:%M:%S'
time_datefmt = '%H:%M:%S'

short_format = logging.Formatter(
    fmt='%(levelname)-8s | %(asctime)s | %(message)s',
    datefmt=time_datefmt
)

console_format = colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)-8s%(reset)s | %(black)s%(asctime)s | %(white)s%(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    datefmt=time_datefmt
)

file_format = logging.Formatter(
    fmt='%(levelname)-8s | %(asctime)s | (%(name)s) - %(message)s',
    datefmt=short_datefmt
)

log_levels = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}

class IOManager:
    def __init__(self, path=None, dir_name="data", log_name = "pta", console_log='info'):
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        if path is None or path == "":
            path = "."

        self.path = pathlib.Path(path).resolve()
        self.local_path = pathlib.Path(__file__).resolve().parent / ".locals"
        self.data_path = self.path / dir_name
        self.storage_path = self.data_path / ".storage"
        
        self.data_path.mkdir(exist_ok=True)
        self.storage_path.mkdir(exist_ok=True)
        
        self.logger = self.make_logger(log_name)
        self.console_handler = self.make_console_handler(self.logger, level=console_log)
        self.file_handler = self.make_file_handler(self.logger, self.data_path)

        for file_path in self.local_path.iterdir():
            if file_path.is_file():
                shutil.copy(file_path, self.storage_path / file_path.name)
        self.logger.debug(f"Copy storage")

    def setup(self, exp_name="test"):
        self.exp_name = exp_name
        self.exp_path = self.data_path / exp_name
        self.img_path = self.exp_path / "img"
        self.psr_path = self.exp_path / "psr"
        self.mod_path = self.psr_path / "mod"
        self.obs_path = self.psr_path / "obs"
        self.gwb_path = self.psr_path / "gwb"

        self.exp_path.mkdir(exist_ok=True) 
        self.img_path.mkdir(exist_ok=True) 
        self.psr_path.mkdir(exist_ok=True) 
        self.mod_path.mkdir(exist_ok=True)
        self.obs_path.mkdir(exist_ok=True)
        self.gwb_path.mkdir(exist_ok=True)

        self.exp_logger = self.make_logger(f"{self.logger.name}.{exp_name}")
        self.exp_handler = self.make_file_handler(
            self.exp_logger, 
            self.exp_path, 
            log_format=short_format, 
            mode='w'
        )

    def make_logger(self, log_name, level=logging.DEBUG):
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)
        return logger

    def make_console_handler(self, logger, log_format=console_format, level='info'):
        console_handler = colorlog.StreamHandler()
        console_handler.setLevel(log_levels[level])
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        return console_handler

    def make_file_handler(self, logger, path, log_format=file_format, level='debug', mode='a'):
        file_handler = logging.FileHandler(path / f"{logger.name}.log", encoding='utf-8', mode=mode)
        file_handler.setLevel(log_levels[level])
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        return file_handler

    def debug(self, message):
        self.exp_logger.debug(message)

    def info(self, message):
        self.exp_logger.info(message)

    def warning(self, message):
        self.exp_logger.warning(message)

    def error(self, message):
        self.exp_logger.error(message)

    def critical(self, message):
        self.exp_logger.critical(message)

    def open_atnf(self):
        atnf_df = pd.read_pickle(self.storage_path / "atnf.pkl")
        self.logger.debug("Open Full ATNF Pulsar Catalog")
        return atnf_df

    def open_pta_catalog(self):
        psrcat_df = pd.read_pickle(self.storage_path / "pta_cat.pkl")
        self.logger.debug("Open PTA Pulsar Catalog")
        return psrcat_df

    def open_atnf_catalog(self):
        psrcat_df = pd.read_pickle(self.storage_path / "atnf_cat.pkl")
        self.logger.debug("Open ATNF Pulsar Catalog")
        return psrcat_df

    def open_psr_catalog(self):
        psrcat = np.genfromtxt(self.storage_path / "psrcat.txt", skip_header=1, dtype="str", unpack=True)
        psrcat_num = np.zeros(psrcat.shape)
        for k in range(len(psrcat)):
            psrcat_num[k] = pd.to_numeric(psrcat[k], errors='coerce')
        return psrcat_num

    def load_atnf(self):
        query = psrqpy.QueryATNF()
        query.pandas.to_pickle(self.storage_path / "atnf.pkl")
        self.logger.debug("Load Full ATNF Pulsar Catalog")

    def load_atnf_catalog(self):
        self.load_atnf()
        atnf = self.open_atnf()
        pta_cat = self.open_pta_catalog()
        atnf_cat = atnf[atnf['PSRJ'].isin(pta_cat['PSRJ'])].reset_index()[pta_cat.columns]
        atnf_cat.to_pickle(self.storage_path / "atnf_cat.pkl")
        self.logger.info("Load ATNF Pulsar Catalog")

    def load_catalog(self, name, path=None):
        cat_name = f"{name}_cat.pkl"
        if path is None:
            path = self.exp_path
        psrcat_df = pd.read_pickle(f"{path}/{cat_name}")
        self.exp_logger.info(f"Load {name} Pulsar Catalog")
        return psrcat_df

    def load_model_psr(self, model_psr='J0030+0451'):
        urlretrieve(f"{model_psr_source}/{model_psr}.par", str(self.storage_path / f"{model_psr}.par"))
        urlretrieve(f"{model_psr_source}/{model_psr}.tim", str(self.storage_path / f"{model_psr}.tim"))
        self.logger.info("Load Model Pulsar")

    def check_model_psr(self, model_psr='J0030+0451'):
        exist_par = os.path.exists(self.storage_path / f"{model_psr}.par")
        exist_tim = os.path.exists(self.storage_path / f"{model_psr}.tim")
        if not exist_par or not exist_tim:
            self.load_model_psr(model_psr)

    def save_catalog(self, catalog):
        catalog.to_pickle(self.exp_path/f"{catalog.name}_cat.pkl")
        self.logger.info(f"Save {catalog.name} Pulsar Catalog")
