# birds_dataset.py
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from constants import IMG_SIZE, BATCH_SIZE, DATA_ROOT_DIR

class BirdsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filepath = self.df.iloc[idx]['filepath']
        label_idx = self.df.iloc[idx]['label_idx']

        img = cv2.imread(filepath, cv2.IMREAD_COLOR)

        if img is None:
            print(f"Warning: Could not read image {filepath}. Returning dummy data.")
            # Return dummy data for robustness or raise an error
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.tensor(0, dtype=torch.long)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        from PIL import Image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label_idx, dtype=torch.long)


def get_dataloaders(csv_file='image_data.csv', test_size=0.15, val_size=0.15, random_state=42):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"'{csv_file}' not found. Please run 'data_preparation.py' first.")

    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError(f"'{csv_file}' is empty. Ensure '{DATA_ROOT_DIR}' has images.")

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label_idx'])
    train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=random_state, stratify=train_df['label_idx'])

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(), # PIL IMAGE (H, W, C) to tensor(C, H, W)
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_dataset = BirdsDataset(train_df, transform=train_transform)
    val_dataset = BirdsDataset(val_df, transform=val_test_transform)
    test_dataset = BirdsDataset(test_df, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()//2 or 1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()//2 or 1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()//2 or 1, pin_memory=True)

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    return train_loader, val_loader, test_loader
