import os
from typing import Any, Callable, Dict, Optional

import pandas as pd
import cv2
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import rasterio

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .transforms import get_pretrained_s2_train_transform, get_s2_train_transform, mv_noaugment

import warnings
from rasterio.errors import NotGeoreferencedWarning
# Suppress the NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

CHECK_MIN_FILESIZE = 20000 # 10kb

class S2GeoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        datapoints_csv,
        data_dir_GS,
        data_dir_OSM,
        data_dir_S2,
        data_dir_FM,
        batch_size: int,
        num_workers: int,
        crop_size: int = 224,
        val_random_split_fraction: float = 0.1,
        transform: str = 'mv_noaugment', # others can be used for testing stuff
        mode: str = "both",
    ):
        super().__init__()

        self.data_dir = data_dir
        self.datapoints_csv = datapoints_csv
        self.data_dir_GS = data_dir_GS
        self.data_dir_OSM = data_dir_OSM
        self.data_dir_S2 = data_dir_S2
        self.data_dir_FM = data_dir_FM
        self.batch_size = batch_size
        self.num_workers = num_workers
        if transform=='pretrained':
            self.train_transform = get_pretrained_s2_train_transform(resize_crop_size=crop_size)
        elif transform=='default':
            self.train_transform = get_s2_train_transform(resize_crop_size=crop_size)
        elif transform=='mv_noaugment':
            self.train_transform = mv_noaugment(resize_crop_size=crop_size)
        else:
            self.train_transform = transform
        self.val_random_split_fraction = val_random_split_fraction
        self.mode = mode
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        if self.data_dir_GS != 'None' and not os.path.exists(self.data_dir_GS):
            print(f"No Google Satellite dataset found in the data directory: {self.data_dir_GS}")
        if self.data_dir_OSM != 'None' and not os.path.exists(self.data_dir_OSM):
            print(f"No OSM dataset found in the data directory: {self.data_dir_OSM}")
        if self.data_dir_S2 != 'None' and not os.path.exists(self.data_dir_S2):
            print(f"No Sentinel2 dataset found in the data directory: {self.data_dir_S2}")
        if self.data_dir_FM != 'None' and not os.path.exists(self.data_dir_FM):
            print(f"No floodmaps dataset found in the data directory: {self.data_dir_FM}")
        if not os.path.exists(self.datapoints_csv):
            print(f"No datapoints CSV found at location: {self.datapoints_csv}")

    def setup(self, stage="fit"):
        dataset = S2Geo(root=self.data_dir,
                        datapoints_csv=self.datapoints_csv, 
                        root_GS=self.data_dir_GS,
                        root_OSM=self.data_dir_OSM,
                        root_S2=self.data_dir_S2,
                        root_FM=self.data_dir_FM,
                        transform=self.train_transform, mode=self.mode)

        N_val = int(len(dataset) * self.val_random_split_fraction)
        N_train = len(dataset) - N_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [N_train, N_val])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        raise NotImplementedError

class S2Geo(NonGeoDataset):

    def __init__(
        self,
        root,
        datapoints_csv,
        root_GS,
        root_OSM,
        root_S2,
        root_FM,
        transform: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = 'default',
        mode: Optional[str] = "both",
    ) -> None:
        assert mode in ["both", "points"]
        self.root = root
        self.datapoints_csv = datapoints_csv
        self.root_GS = root_GS
        self.root_OSM = root_OSM
        self.root_S2 = root_S2
        self.root_FM = root_FM
        self.transform = transform
        self.mode = mode

        df = pd.read_csv(self.datapoints_csv)

        self.gs_files = []
        self.osm_files = []
        self.s2_files = []
        self.fm_files = []
        self.points = []

        n_skipped_files = 0
        for i in range(df.shape[0]):

            gs_file = os.path.join(
                self.root_GS,
                str(df.iloc[i]["datapoint_id"]) + ".png"
            )
            if os.path.exists(gs_file) and os.path.getsize(gs_file) < 10000:
                n_skipped_files += 1
                continue
            self.gs_files.append(gs_file)

            osm_file = os.path.join(
                self.root_OSM,
                str(df.iloc[i]["datapoint_id"]) + ".pkl"
            )
            self.osm_files.append(osm_file)

            s2_file = os.path.join(
                self.root_S2,
                str(df.iloc[i]["datapoint_id"]) + ".tiff"
            )
            if os.path.exists(s2_file) and os.path.getsize(s2_file) < 20000:
                n_skipped_files += 1
                continue
            self.s2_files.append(s2_file)

            fm_file = os.path.join(
                self.root_FM,
                str(df.iloc[i]["datapoint_id"]) + ".png"
            )
            self.fm_files.append(fm_file)

            self.points.append([df.iloc[i]["longitude"], df.iloc[i]["latitude"]])

        print(f"skipped {n_skipped_files}/{len(df)} images because they were smaller "
              f"than {CHECK_MIN_FILESIZE} bytes... they probably contained nodata pixels")

    def __getitem__(self, index: int) -> Dict[str, Tensor]:

        point = torch.tensor(self.points[index])
        sample = {"point": point}

        sample["gs"] = None
        sample["osm"] = None
        sample["s2"] = None
        sample["fm"] = None

        if self.mode == "both":

            if self.root_GS != 'None':
                gs = cv2.imread(self.gs_files[index])
                gs = cv2.cvtColor(gs, cv2.COLOR_BGR2RGB)
                gs = np.float32(gs)
                sample["gs"] = gs

            if self.root_OSM != 'None':
                with open(self.osm_files[index], 'rb') as f:
                    osm_data = pd.read_pickle(f)
                    sample["osm"] = osm_data

            if self.root_S2 != 'None':
                with rasterio.open(self.s2_files[index]) as f:
                    data = f.read()
                    s2 = data
                sample["s2"] = s2

            if self.root_FM != 'None':
                fm = cv2.imread(self.fm_files[index])
                fm = np.float32(fm)
                sample["fm"] = fm
            
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.
        Returns:
            length of dataset
        """
        return len(self.points)

    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.
        Returns:
            True if the dataset directories and split files are found, else False
        """
        
        for filename in self.validation_filenames:
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                print(filepath +' missing' )
                return False
        return True

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.
        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
        ncols = 1

        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        ax.imshow(image[:, :, [0,1,2]] / 255)
        ax.axis("off")

        if show_titles:
            ax.set_title(f"({sample['point'][0]:0.4f}, {sample['point'][1]:0.4f})")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig