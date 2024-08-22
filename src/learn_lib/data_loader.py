import torch
import lightning as L
import numpy as np
import pandas as pd

from typing import TypeAlias, Generator

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


RGB: TypeAlias = tuple[int, int, int]

IMAGE_SIZE: int = 224


class KenyanFood13Dataset(Dataset):
    """
    KenyanFood13Dataset is Dataset class for the Kenyan Food 13 dataset.
    """

    def __init__(
        self,
        data_root: str,
        train=True,
        transform=lambda x: x,
        random_state: int = 42,
    ) -> None:
        """
        Initializes the dataset.

        Args:
            train (bool): True if training data, False if validation data
        """
        self.data_csv_path = f"{data_root}/train.csv"
        self.images_dir = f"{data_root}/images/images"

        self.train = train
        self.transform = transform

        self.df = pd.read_csv(self.data_csv_path)
        data_np = self.df.to_numpy()
        X = data_np[:, 0]
        y = data_np[:, 1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=random_state,
        )

        if self.train:
            self.X, self.y = self.X_train, self.y_train
        else:
            self.X, self.y = self.X_test, self.y_test

        self.class_to_idx = {c: i for i, c in enumerate(np.unique(y))}

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: length of the dataset
        """
        return len(self.X)

    def __getitem__(self, idx) -> tuple[Image.Image, int] | Image.Image:
        """
        Returns the image and label for training dataset and only the image for validation dataset.

        Args:
            idx (int): index of the item
        Returns:
            tuple[Image.Image, int]: (image, label)
        """
        img_id, img_class = self.X[idx], self.y[idx]
        img = Image.open(f"{self.images_dir}/{img_id}.jpg")

        return self.transform(img), self.class_to_idx[img_class]


class KenyanFood13DataModule(L.LightningDataModule):
    """
    KenyanFood13DataModule is LightningDataModule class for the Kenyan Food 13 dataset.
    """

    def __init__(
        self,
        data_root: str,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        num_workers: int = 4,
        use_augmentation: bool = False,
    ) -> None:
        """
        Initializes KenyanFood13DataModule

        Args:
            data_root (str): path to the data root
            train_batch_size (int, optional): batch size for training. Defaults to 32.
            val_batch_size (int, optional): batch size for validation. Defaults to 32.
            num_workers (int, optional): number of workers. Defaults to 4.
            use_augmentation (bool, optional): whether to use data augmentation. Defaults to False.

        Returns:
            None
        """
        super().__init__()

        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.use_augmentation = use_augmentation

        mean, std = get_mean_and_std(data_root)
        self.common_transforms = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        if use_augmentation:
            self.train_transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=(0, 30)),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomPerspective(
                        distortion_scale=0.2,
                        p=0.5,
                    ),
                    self.common_transforms,
                    transforms.RandomErasing(),
                ]
            )
        else:
            self.train_transforms = self.common_transforms

    def setup(self, stage: str | None = None) -> None:
        """
        Loads the train and validation datasets.

        Args:
            stage (str | None, optional): stage. Defaults to None.

        Returns:
            None
        """
        self.train_dataset = KenyanFood13Dataset(
            data_root=self.data_root,
            train=True,
            transform=self.train_transforms,
        )

        self.val_dataset = KenyanFood13Dataset(
            data_root=self.data_root,
            train=False,
            transform=self.common_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Creates and returns the training dataloader.

        Returns:
            DataLoader: training dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Creates and returns the validation dataloader.

        Returns:
            DataLoader: validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


def get_mean_and_std(data_root: str) -> tuple[RGB, RGB]:
    """
    Computes the mean and standard deviation of the train dataset.

    Returns:
        tuple[RGB, RGB]: (mean, std)
    """
    preprocess_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = KenyanFood13Dataset(
        data_root=data_root,
        train=True,
        transform=preprocess_transforms,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False,
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)
    for imgs, _ in dataloader:
        for i in range(3):
            mean[i] += imgs[:, i, :, :].mean()
            std[i] += imgs[:, i, :, :].std()

    mean.div_(len(dataloader))
    std.div_(len(dataloader))

    return mean, std


def kaggle_test_images(
    data_root: str,
) -> Generator[tuple[torch.Tensor, str], None, None]:
    """
    Gets the test images from the Kaggle dataset.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (image_names, images)
    """
    data_csv_path = f"{data_root}/test.csv"
    images_dir = f"{data_root}/images/images"

    df = pd.read_csv(data_csv_path)
    image_names = df.to_numpy()[:, 0]

    mean, std = get_mean_and_std(data_root)
    common_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    for image_name in image_names:
        image = Image.open(f"{images_dir}/{image_name}.jpg")
        yield common_transforms(image), image_name
