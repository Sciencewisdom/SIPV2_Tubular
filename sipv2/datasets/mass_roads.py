"""
Massachusetts Roads Dataset loader.
150 train / 49 val / 49 test
1500x1500 aerial images with binary road masks.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


class MassachusettsRoadsDataset(Dataset):
    """
    Massachusetts Roads Dataset.

    Expected directory structure:
        data/raw/mass_roads/train/sat/*.tiff
        data/raw/mass_roads/train/map/*.tiff
        data/raw/mass_roads/valid/sat/*.tiff
        data/raw/mass_roads/valid/map/*.tiff
        data/raw/mass_roads/test/sat/*.tiff
        data/raw/mass_roads/test/map/*.tiff
    """

    def __init__(
        self,
        root_dir,
        split='train',
        crop_size=512,
        augment=True,
        normalize=True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.crop_size = crop_size
        self.augment = augment
        self.normalize = normalize

        img_dir = os.path.join(root_dir, split, 'sat')
        gt_dir = os.path.join(root_dir, split, 'map')

        self.image_paths = sorted(glob.glob(os.path.join(img_dir, '*.tif')) + glob.glob(os.path.join(img_dir, '*.tiff')))
        self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.tif')) + glob.glob(os.path.join(gt_dir, '*.tiff')))

        assert len(self.image_paths) > 0, f"No images found in {img_dir}"
        assert len(self.gt_paths) > 0, f"No ground truth found in {gt_dir}"
        assert len(self.image_paths) == len(self.gt_paths), \
            f"Image/GT mismatch: {len(self.image_paths)} vs {len(self.gt_paths)}"

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        """Load image as numpy array [H, W, C]."""
        img = np.array(Image.open(path))
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        return img.astype(np.float32)

    def _load_mask(self, path):
        """Load mask as binary numpy array [H, W]."""
        mask = np.array(Image.open(path))
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        if mask.max() > 1:
            mask = mask / 255.0
        return (mask > 0.5).astype(np.float32)

    def _random_crop(self, img, mask, deterministic=False):
        """Random crop to crop_size. If deterministic, use center crop."""
        h, w = img.shape[:2]
        ch, cw = self.crop_size, self.crop_size
        if h < ch or w < cw:
            # Resize if image is smaller than crop
            img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
            return img, mask

        if deterministic:
            top = (h - ch) // 2
            left = (w - cw) // 2
        else:
            top = np.random.randint(0, h - ch + 1)
            left = np.random.randint(0, w - cw + 1)
        img = img[top:top+ch, left:left+cw]
        mask = mask[top:top+ch, left:left+cw]
        return img, mask

    def _augment(self, img, mask):
        """Data augmentation."""
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        if np.random.rand() > 0.5:
            angle = np.random.uniform(-30, 30)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if new_h > h or new_w > w:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                img = img[start_h:start_h + h, start_w:start_w + w]
                mask = mask[start_h:start_h + h, start_w:start_w + w]
            else:
                pad_h = h - new_h
                pad_w = w - new_w
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Brightness / Contrast
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-20, 20)
            img = np.clip(alpha * img + beta, 0, 255)

        # Gaussian noise
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 5, img.shape)
            img = np.clip(img + noise, 0, 255)

        return img, mask

    def __getitem__(self, idx):
        img = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.gt_paths[idx])

        # Crop (center crop for validation, random for train)
        img, mask = self._random_crop(img, mask, deterministic=not self.augment)

        # Augment
        if self.augment:
            img, mask = self._augment(img, mask)

        # Normalize
        img = img / 255.0

        # To tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            'image': img,
            'mask': mask,
            'fov_mask': torch.ones_like(mask),  # dummy fov mask for compatibility
            'image_id': os.path.basename(self.image_paths[idx]).split('.')[0],
        }


def get_mass_roads_loaders(
    root_dir,
    crop_size=512,
    batch_size=2,
    num_workers=2,
):
    """Create train and validation dataloaders."""
    train_dataset = MassachusettsRoadsDataset(
        root_dir=root_dir,
        split='train',
        crop_size=crop_size,
        augment=True,
    )
    val_dataset = MassachusettsRoadsDataset(
        root_dir=root_dir,
        split='valid',
        crop_size=crop_size,
        augment=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
