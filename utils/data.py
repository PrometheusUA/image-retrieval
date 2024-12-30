import os
import random

import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, ConcatDataset
from torchvision import transforms as T
from PIL import Image, ImageFilter

from utils.constants import TRAIN_CSV, TEST_CSV, TRAIN_DIR, TEST_DIR


class TwoPacksTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return self.transforms(x), self.transforms(x)


class TwoTransformsParallelTransform:
    def __init__(self, transforms1, transforms2):
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, x):
        return self.transforms1(x), self.transforms2(x)


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x



class ImageDataset(Dataset):
    def __init__(self, file_ids, images_folder=None, classes=None, transforms=None):
        self.file_ids = file_ids
        self.images_folder = images_folder
        self.classes = classes
        self.transforms = transforms

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        image_path = f"{file_id}.jpg"
        if self.images_folder is not None:
            image_path = os.path.join(self.images_folder, image_path)
        img = Image.open(image_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        if self.classes is not None:
            return img, self.classes[idx]
        return img


def get_default_transforms(dataset_type, image_size=224):
    if dataset_type == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.ToTensor(),
        ])
    elif dataset_type == "test":
        return T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])
    else:
        raise NotImplementedError(f"Unknown dataset type: {dataset_type}")


class CombinedLoader:
    def __init__(self, labeled_loader, unlabeled_loader):
        """
        Combined loader that works on an epoch-level.
        It aligns batches of labeled and unlabeled datasets.

        Args:
            labeled_loader: DataLoader for labeled data.
            unlabeled_loader: DataLoader for unlabeled data.
        """
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader

    def __iter__(self):
        """
        Create an iterator that combines labeled and unlabeled batches.
        Shortens to the smaller loader's length to ensure alignment.
        """
        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)
        for labeled_batch, unlabeled_batch in zip(labeled_iter, unlabeled_iter):
            yield labeled_batch, unlabeled_batch

    def __len__(self):
        """
        Return the number of batches per epoch, determined by the smaller loader.
        """
        return min(len(self.labeled_loader), len(self.unlabeled_loader))


def get_loaders(
        train_csv_path=TRAIN_CSV, 
        test_csv_path=TEST_CSV,
        train_files_dir=TRAIN_DIR,
        test_files_dir=TEST_DIR,
        train_transforms=None,
        test_transforms=None,
        train_on_test_too=False,
        semisupervised=False,
        val_frac=0.05,
        batch_size=32,
        num_workers=1,
        persistent_workers=False,
        train_shuffle=True,
        pin_memory=False):
    assert not (train_on_test_too and semisupervised), "Can't train on test data in semisupervised mode"

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    if semisupervised:
        labeled_dataset = ImageDataset(
            file_ids=train_df["file_id"].values,
            images_folder=train_files_dir,
            classes=train_df["class"].values,
            transforms=train_transforms
        )

        unlabeled_dataset = ImageDataset(
            file_ids=test_df["file_id"].values,
            images_folder=test_files_dir,
            classes=None,
            transforms=train_transforms
        )
        
        val_classes_ids = train_df['class'].value_counts().sample(frac=3*val_frac).index
        val_samples_ids = train_df.loc[train_df['class'].isin(val_classes_ids)].groupby("class").sample(n=1).index
        train_samples_ids = train_df.index.difference(val_samples_ids)
        train_sampler_labeled = SubsetRandomSampler(train_samples_ids.tolist())
        val_sampler_labeled = SubsetRandomSampler(val_samples_ids.tolist())

        val_samples_ids_unlabeled = test_df.sample(frac=val_frac).index
        train_samples_ids_unlabeled = test_df.index.difference(val_samples_ids_unlabeled)
        train_sampler_unlabeled = SubsetRandomSampler(train_samples_ids_unlabeled.tolist())
        val_sampler_unlabeled = SubsetRandomSampler(val_samples_ids_unlabeled.tolist())

        train_loader_labeled = DataLoader(labeled_dataset, sampler=train_sampler_labeled,
                                            batch_size=batch_size, drop_last=False, num_workers=num_workers//2, 
                                            persistent_workers=persistent_workers, pin_memory=pin_memory)
        
        val_loader_labeled = DataLoader(labeled_dataset, sampler=val_sampler_labeled,
                                batch_size=batch_size, drop_last=False, num_workers=num_workers // 2, 
                                persistent_workers=persistent_workers, pin_memory=pin_memory)
        
        train_loader_unlabeled = DataLoader(unlabeled_dataset, sampler=train_sampler_unlabeled,
                                            batch_size=batch_size, drop_last=False, num_workers=num_workers//2, 
                                            persistent_workers=persistent_workers, pin_memory=pin_memory)
        
        val_loader_unlabeled = DataLoader(unlabeled_dataset, sampler=val_sampler_unlabeled,
                                            batch_size=batch_size, drop_last=False, num_workers=num_workers//2, 
                                            persistent_workers=persistent_workers, pin_memory=pin_memory)
        
        train_loader = CombinedLoader(train_loader_labeled, train_loader_unlabeled)
        val_loader = CombinedLoader(val_loader_labeled, val_loader_unlabeled)

        test_dataset = ImageDataset(
            file_ids=test_df["file_id"].values,
            images_folder=test_files_dir,
            classes=None,
            transforms=test_transforms
        )

        test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             drop_last=False,
                             shuffle=False,
                             num_workers=num_workers,
                             persistent_workers=persistent_workers,
                             pin_memory=pin_memory)
        
        return train_loader, val_loader, test_loader

    train_dataset = ImageDataset(
        file_ids=train_df["file_id"].values,
        images_folder=train_files_dir,
        classes=train_df["class"].values,
        transforms=train_transforms
    )

    test_dataset = ImageDataset(
        file_ids=test_df["file_id"].values,
        images_folder=test_files_dir,
        classes=None,
        transforms=test_transforms
    )

    if val_frac > 1/3:
        raise NotImplementedError("Validation fraction is too large for now")
    
    val_classes_ids = train_df['class'].value_counts().sample(frac=3*val_frac).index
    val_samples_ids = train_df.loc[train_df['class'].isin(val_classes_ids)].groupby("class").sample(n=1).index
    train_samples_ids = train_df.index.difference(val_samples_ids)
    train_sampler = SubsetRandomSampler(train_samples_ids.tolist())
    val_sampler = SubsetRandomSampler(val_samples_ids.tolist())

    if train_on_test_too:
        test_dataset2 = ImageDataset(
            file_ids=test_df["file_id"].values,
            images_folder=test_files_dir,
            classes=[-1] * len(test_df),
            transforms=train_transforms
        )

        old_train_dataset_size = len(train_dataset)

        train_dataset = ConcatDataset([train_dataset, test_dataset2])

        train_sampler = SubsetRandomSampler(train_samples_ids.tolist() + 
                                            list(range(old_train_dataset_size, old_train_dataset_size + len(test_dataset2))))

    train_loader = DataLoader(train_dataset, 
                              sampler=train_sampler, 
                              batch_size=batch_size, 
                              drop_last=False,
                              num_workers=num_workers,
                              persistent_workers=persistent_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(train_dataset, 
                            sampler=val_sampler, 
                            batch_size=batch_size, 
                            drop_last=False,
                            num_workers=num_workers,
                            persistent_workers=persistent_workers,
                            pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             drop_last=False,
                             num_workers=num_workers,
                             persistent_workers=persistent_workers,
                             pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
