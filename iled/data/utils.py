import h5py
import os
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset
from urllib.request import urlretrieve
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """
    Snippet stolen from https://github.com/Extrality/airfrans_lib/blob/main/src/airfrans/dataset.py
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b = 1, bsize = 1, tsize = None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b*bsize - self.n) # also sets self.n = b * bsize

class HDF5Dataset(Dataset):

    def __init__(self, path, split):

        self.data = h5py.File(osp.join(path, split, "data.h5"), "r")
        self.keys = list(self.data.keys())

    def __len__(self):

        return len(self.data.keys())

    def __getitem__(self, idx):

        return torch.tensor(self.data[self.keys[idx]]["data"][:])


class FHNDataset(HDF5Dataset):
    """
    Class defining pre-computed values for the Fitz-Hugh Nagomo data available at:
    https://zenodo.org/records/7813903

    The data (1.5G) will be automatically downloaded if not available.
    """

    def __init__(self, path, split, *args, **kwargs):

        self.dt = 1.0
        self.data_min = np.array(
            [-1.022019956159691212, -0.1749421447666666685]
        ).reshape(-1, 1)
        self.data_max = np.array([1.023366826822246001, 0.1866165320999999822]).reshape(
            -1, 1
        )

        if osp.exists(osp.join(path,split)) is False:
            print(f'Downloading data at {path}')
            self.download(path)

        super().__init__(path,split,*args, **kwargs)

    def download(self,path):
        if not osp.exists(path):
            os.mkdir(path)

        file_name = 'fhn_compressed'
        url = 'https://zenodo.org/records/7813903/files/FHN_data.zip?download=1'
        with DownloadProgressBar(unit = 'B', unit_scale = True, miniters = 1, unit_divisor = 1024, desc = 'Downloading FHN data') as t:
            urlretrieve(url, filename = osp.join(path, file_name + '.zip'), reporthook = t.update_to)

        print("Extracting " + file_name + ".zip at " + path + "...")
        with zipfile.ZipFile(osp.join(path, file_name + '.zip'), 'r') as zipf:
            zipf.extractall(path)

        os.remove(osp.join(path, file_name + '.zip'))