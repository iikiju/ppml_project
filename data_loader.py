import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np
import os

def min_max_normalize(image, min_val=None, max_val=None):
    min_val = image.min()
    max_val = image.max()
    image = (image - min_val) / (max_val - min_val)
    image = image.astype(np.float32)
    return image

def get_transforms(augment=False):
    if augment:
        return A.Compose([
            A.HorizontalFlip(p=0.5),  # 기본적으로 적용할 수 있음

            A.OneOf([
                A.Compose([
                    A.Lambda(image=lambda x, **kwargs: (x * 255).astype(np.uint8), always_apply=True),
                    A.CLAHE(clip_limit=2.0, p=1.0),
                    A.Lambda(image=lambda x, **kwargs: x.astype(np.float32) / 255., always_apply=True)
                ]),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.8),  # 둘 중 하나를 80% 확률로 선택 적용

            ToTensorV2()
        ])
    else:
        return A.Compose([
            ToTensorV2()
        ])

class LIDCDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mask_base_path = "/data1/alice/classifier/image_data/data/Mask"
        image_base_path = "/data1/alice/classifier/image_data/data/Original"
        dicom_img_path = str(row['dicom_img_path'])

        if dicom_img_path == 'nan' or dicom_img_path.strip() == '':
            print(row)
            raise ValueError(f"dicom_img_path is missing for index {idx}")
        image = np.load(os.path.join(image_base_path, dicom_img_path + '.npy'))
        image = min_max_normalize(image)
        image = np.expand_dims(image, axis=-1)
        if self.transform:
            image = self.transform(image=image)["image"]
        label = int(row['malignancy'] >= 1)  # Binary classification
        return image, label

    def __len__(self):
        return len(self.df)

def load_data(csv_path, split='train', client_id=None, augment=False):
    df = pd.read_csv(csv_path)
    
    # Filter by split
    df = df[df['split'] == split]
    
    # If client_id is specified and this is not test data
    if client_id and split != 'test':
        # Check if the client_id column exists and has values
        if 'client_id' not in df.columns:
            raise ValueError(f"Client ID column not found in CSV for {split} split")
        
        # Filter by client_id
        client_df = df[df['client_id'] == client_id]
        
        # Check if we have data for this client
        if len(client_df) == 0:
            print(f"⚠️ Warning: No data found for client {client_id} in {split} split")
            # Return a very small subset just to avoid crashes - you might want to handle this differently
            client_df = df.head(1) if len(df) > 0 else df
        
        df = client_df
    
    transform = get_transforms(augment)
    return LIDCDataset(df, transform)