import pytorch_lightning as pl
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader, random_split
import torch
from typing import Tuple
from dataloader.DDFF12.loader import DDFF12Loader
from .NYU.loader import NYULoader
from dataloader.ARKitScenes.loader import ARKitScenesLoader
from dataloader.smartphone.loader import SmartphoneLoader
from dataloader.BBBC006.loader import BBBC006Loader # Added for BBBC006 dataset


class DDFF12DataModule(pl.LightningDataModule):
    def __init__(
        self,
        ddff12_data_root: str = "",
        img_size: Tuple = (480, 640),
        remove_white_border: bool = True,
        batch_size: int = 32,
        num_workers: int = 16,
        use_labels: bool = True,
        num_cluster: int = 5,
    ):
        """Initialize the data module."""
        super().__init__()
        self.database_root = ddff12_data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_labels = use_labels
        self.num_cluster = num_cluster

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = DDFF12Loader(
                self.database_root,
                stack_key="stack_train",
                disp_key="disp_train",
                n_stack=5,
                min_disp=0.02,
                max_disp=0.28,
            )
            self.valid_dataset = DDFF12Loader(
                self.database_root,
                stack_key="stack_val",
                disp_key="disp_val",
                n_stack=5,
                min_disp=0.02,
                max_disp=0.28,
                b_test=True,
            )
        if stage == "test":
            self.valid_dataset = DDFF12Loader(
                self.database_root,
                stack_key="stack_val",
                disp_key="disp_val",
                n_stack=5,
                min_disp=0.02,
                max_disp=0.28,
                b_test=True,
            )
            self.test_dataset = self.valid_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=1, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

class SmartphoneDataModule(pl.LightningDataModule):
    def __init__(
        self,
        smartphoneRoot: str = "",
        batch_size: int = 1,
        num_workers: int = 16,
        use_labels: bool = True,
        num_cluster: int = 5,
    ):
        """Initialize the data module."""
        super().__init__()
        self.database_root = smartphoneRoot
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_labels = use_labels
        self.num_cluster = num_cluster

    def setup(self, stage: str):
        if stage == "test":
            self.valid_dataset = SmartphoneLoader(self.database_root)
            self.test_dataset = self.valid_dataset

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=1, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

class NYUDataModule(pl.LightningDataModule):
    def __init__(
        self,
        nyuv2_data_root: str = "",
        img_size: Tuple = (480, 640),
        remove_white_border: bool = True,
        batch_size: int = 32,
        num_workers: int = 16,
        use_labels: bool = True,
        num_cluster: int = 5,
        n_stack: int = 10,
    ):
        """Initialize the data module."""
        super().__init__()
        # Set the seed for PyTorch
        # torch.manual_seed(42)
        self.nyu_data_root = nyuv2_data_root
        self.image_size = img_size
        self.remove_white_border = remove_white_border
        self.n_stack = n_stack
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_labels = use_labels
        self.num_cluster = num_cluster

        # dataset_train, dataset_valid = random_split(dataset, [0.9, 0.1])

        # self.train_dataset = dataset_train
        # self.valid_dataset = dataset_valid
        # self.test_dataset = dataset_valid

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = NYULoader(
                self.nyu_data_root,
                img_size=self.image_size,
                remove_white_border=self.remove_white_border,
                n_stack=self.n_stack,
                stage="train"
            )
            self.test_dataset = NYULoader(
                self.nyu_data_root,
                img_size=self.image_size,
                remove_white_border=self.remove_white_border,
                n_stack=self.n_stack,
                stage="test"
            )
        if stage == "test":
            self.test_dataset = NYULoader(
                self.nyu_data_root,
                img_size=self.image_size,
                remove_white_border=self.remove_white_border,
                n_stack=self.n_stack,
                stage="test"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=1, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

class ARKitScenesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        arkitScenes_data_root: str = "",
        img_size: Tuple = (192, 256),
        batch_size: int = 32,
        num_workers: int = 16,
        num_cluster: int = 5,
    ):
        """Initialize the data module."""
        super().__init__()
        # Set the seed for PyTorch
        # torch.manual_seed(42)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_cluster = num_cluster

        self.arkitScenes_data_root = arkitScenes_data_root
        self.image_size = img_size

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = ARKitScenesLoader(
                ARKitScenes_data_root=self.arkitScenes_data_root,
                img_size=self.image_size,
                stage="Training"
            )
            self.valid_dataset = ARKitScenesLoader(
                ARKitScenes_data_root=self.arkitScenes_data_root,
                img_size=self.image_size,
                stage="Validation"
            )
        if stage == "test":
            self.test_dataset = ARKitScenesLoader(
                ARKitScenes_data_root=self.arkitScenes_data_root,
                img_size=self.image_size,
                stage="Validation"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=1, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

#
# ========================================================================
#   NEW DATAMODULE FOR BBBC006
# ========================================================================
#
class BBBC006DataModule(pl.LightningDataModule):
    def __init__(
        self,
        bbbc006_data_root: str = "",
        img_size: Tuple = (512, 512),
        batch_size: int = 32,
        num_workers: int = 16,
        n_stack: int = 5,
        z_step_um: float = 2.0,
        val_split: float = 0.1,
        use_labels: bool = True,
        num_cluster: int = 5,
    ):
        """Initialize the data module."""
        super().__init__()
        self.database_root = bbbc006_data_root
        self.image_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_stack = n_stack
        self.z_step_um = z_step_um
        self.val_split = val_split
        self.use_labels = use_labels
        self.num_cluster = num_cluster
        
        # These will be populated in setup()
        self.full_dataset = None
        self.train_split = None
        self.val_split_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        
        # setup is called per-process, so we can create dataset instances here
        
        if stage == "fit":
            self.full_dataset = BBBC006Loader(
                root_dir=self.database_root,
                img_size=self.image_size,
                n_stack=self.n_stack,
                z_step_um=self.z_step_um
            )
            
            # Split the dataset into train and validation
            val_size = int(len(self.full_dataset) * self.val_split)
            train_size = len(self.full_dataset) - val_size
            
            self.train_split, self.val_split_dataset = random_split(
                self.full_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42) # for reproducibility
            )
            
        if stage == "test":
            self.test_dataset = BBBC006Loader(
                root_dir=self.database_root,
                img_size=self.image_size,
                n_stack=self.n_stack,
                z_step_um=self.z_step_um
            )
            # Set the test dataset to 'test' mode (fixed slices)
            self.test_dataset.set_stage("test")


    def train_dataloader(self):
        # Set the underlying dataset to 'train' mode (random slices)
        # self.train_split is a Subset, .dataset accesses the original BBBC006Loader
        self.train_split.dataset.set_stage("train")
        return DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        # Set the underlying dataset to 'val' mode (fixed slices)
        self.val_split_dataset.dataset.set_stage("val")
        return DataLoader(
            self.val_split_dataset, 
            batch_size=1, # Often 1 for validation
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=1, 
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=1, 
            num_workers=self.num_workers
        )