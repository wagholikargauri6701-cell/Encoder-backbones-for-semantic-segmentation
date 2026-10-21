import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

NUM_CLASSES  = 11
IGNORE_INDEX = 255

CLASS_NAMES_11 = [
    "Sky", "Building", "ColumnPole", "Road", "Sidewalk",
    "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist",
]

# Maps CamVid RGB colours to 11 class indices.
# Anything not listed here stays at IGNORE_INDEX (void).
CAMVID_COLORS = {
    (128, 128, 128): 0,   # Sky

    (128,   0,   0): 1,   # Building
    (192,   0, 128): 1,
    ( 64,   0,  64): 1,
    ( 64, 192,   0): 1,
    (  0, 128,  64): 1,

    (192, 192, 128): 2,   # ColumnPole

    (128,  64, 128): 3,   # Road
    (128,   0, 192): 3,
    (192,   0,  64): 3,
    ( 64, 192, 128): 3,
    (128, 128, 192): 3,

    (  0,   0, 192): 4,   # Sidewalk

    (128, 128,   0): 5,   # Tree
    (192, 192,   0): 5,

    (192, 128, 128): 6,   # SignSymbol
    (  0,  64,  64): 6,
    (  0,   0,  64): 6,

    ( 64,  64, 128): 7,   # Fence

    ( 64,   0, 128): 8,   # Car
    ( 64, 128, 192): 8,
    (192, 128, 192): 8,
    (192,  64, 128): 8,
    (192,   0, 192): 8,
    (128,  64,  64): 8,
    ( 64,   0, 192): 8,

    ( 64,  64,   0): 9,   # Pedestrian
    (192, 128,  64): 9,
    ( 64, 128,  64): 9,

    (  0, 128, 192): 10,  # Bicyclist

    (128, 128,  64): IGNORE_INDEX,  # void
    (  0,   0,   0): IGNORE_INDEX,
}

class CamVidDataset(Dataset):

    def __init__(self, root, split="train", img_size=256, augment=False):
        self.root    = root
        self.split   = split
        self.img_size = img_size
        self.augment = augment

        self.img_dir = os.path.join(root, split)
        self.lbl_dir = os.path.join(root, f"{split}_labels")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image folder not found: {self.img_dir}")
        if not os.path.isdir(self.lbl_dir):
            raise FileNotFoundError(f"Label folder not found: {self.lbl_dir}")

        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if len(self.images) == 0:
            raise RuntimeError("No images found.")

        # Pre-build a flat 256^3 LUT so RGB->class is one vectorised index op,
        self.mask_lut = np.full(256 ** 3, IGNORE_INDEX, dtype=np.int16)
        for color, cls in CAMVID_COLORS.items():
            idx = color[0] * 65536 + color[1] * 256 + color[2]
            self.mask_lut[idx] = cls

        # ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        self.train_transform = A.Compose([
            #resize
            A.Resize(img_size, img_size,
                     interpolation=cv2.INTER_LINEAR,
                     mask_interpolation=cv2.INTER_NEAREST),
                     #horizontal flip
            A.HorizontalFlip(p=0.5),
            #rotate +-10 degrees 
            A.Rotate(limit=10,
                     interpolation=cv2.INTER_LINEAR,
                     mask_interpolation=cv2.INTER_NEAREST,
                     border_mode=cv2.BORDER_CONSTANT,
                     fill=0,
                     fill_mask=IGNORE_INDEX,
                     p=0.5),
            #random scaling
            A.RandomScale(scale_limit=0.2,
                          interpolation=cv2.INTER_LINEAR,
                          mask_interpolation=cv2.INTER_NEAREST,
                          p=0.5),
            #padding if scalling applies and image is too small
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT,
                          fill=0,
                          fill_mask=IGNORE_INDEX),
            #radom crop
            A.RandomCrop(height=img_size, width=img_size),
            # Safety resize — ensures exact shape after scale+pad+crop
            A.Resize(img_size, img_size,
                     interpolation=cv2.INTER_LINEAR,
                     mask_interpolation=cv2.INTER_NEAREST),
            #brightness, contrast and colour changes for lighting
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5),
            
            #different exposure conditions    
            A.RandomGamma(gamma_limit=(70, 140), p=0.3),
            #blur effect
            A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(0.1, 2.0), p=0.3),
            #noise to improve generalisation
            A.GaussNoise(std_range=(0.025, 0.23), p=0.3),
            
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

        self.val_transform = A.Compose([
            A.Resize(img_size, img_size,
                     interpolation=cv2.INTER_LINEAR,
                     mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        img_path  = os.path.join(self.img_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + "_L.png"
        mask_path = os.path.join(self.lbl_dir, mask_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(mask_path)

        image    = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

        # Convert RGB mask to class indices before any spatial transform
        # so interpolation never blends two class colours into an invalid RGB.
        lut_index = (
            mask_rgb[..., 0].astype(np.int32) * 65536
            + mask_rgb[..., 1].astype(np.int32) * 256
            + mask_rgb[..., 2].astype(np.int32)
        )
        mask = self.mask_lut[lut_index].astype(np.uint8)

        transform = self.train_transform if self.augment else self.val_transform
        out = transform(image=image, mask=mask)

        image = out["image"]
        mask  = out["mask"].long()
        return image, mask