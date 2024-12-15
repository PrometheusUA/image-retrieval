import os

import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms as T
from PIL import Image

from utils.constants import TRAIN_CSV, TEST_CSV, TRAIN_DIR, TEST_DIR


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


def get_default_transforms(dataset_type):
    if dataset_type == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.ToTensor(),
        ])
    elif dataset_type == "test":
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
    else:
        raise NotImplementedError(f"Unknown dataset type: {dataset_type}")


def get_loaders(
        train_csv_path=TRAIN_CSV, 
        test_csv_path=TEST_CSV,
        train_files_dir=TRAIN_DIR,
        test_files_dir=TEST_DIR,
        train_transforms=None,
        test_transforms=None,
        val_frac=0.05,
        batch_size=32,
        num_workers=1,
        persistent_workers=False,
        pin_memory=False):
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

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
