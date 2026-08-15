"""
DRIVE dataset loader for retinal vessel segmentation.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


class DRIVEDataset(Dataset):
    """
    DRIVE dataset.

    Expected directory structure:
        data/raw/DRIVE/training/images/*.tif
        data/raw/DRIVE/training/1st_manual/*.gif
        data/raw/DRIVE/training/mask/*.gif
        data/raw/DRIVE/test/images/*.tif
        data/raw/DRIVE/test/1st_manual/*.gif
        data/raw/DRIVE/test/mask/*.gif
        data/raw/DRIVE/test/2nd_manual/*.gif
    """

    def __init__(
        self,
        root_dir,
        split='train',
        img_size=512,
        augment=True,
        normalize=True,
        val_indices=None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.normalize = normalize

        # DRIVE test set often doesn't have public ground truth.
        # For train/val, we ALWAYS split from the training set.
        # Only use test set if it has ground truth AND split == 'test'.
        use_test = (split == 'test')
        if use_test:
            # Check if test set has GT
            test_img_dir = os.path.join(root_dir, 'test', 'images')
            test_gt_dir = os.path.join(root_dir, 'test', '1st_manual')
            test_img_paths = sorted(glob.glob(os.path.join(test_img_dir, '*.tif')))
            test_gt_paths = sorted(glob.glob(os.path.join(test_gt_dir, '*.gif')))
            if len(test_gt_paths) > 0 and len(test_gt_paths) == len(test_img_paths):
                # Test set has GT, use it
                img_dir = test_img_dir
                gt_dir = test_gt_dir
                mask_dir = os.path.join(root_dir, 'test', 'mask')
            else:
                use_test = False
                import warnings
                warnings.warn(
                    f"DRIVE test split has no GT under {test_gt_dir}; "
                    "falling back to the 4-case validation split held out from training"
                )

        if not use_test:
            # Use training set with fixed split
            split_dir = 'training'
            img_dir = os.path.join(root_dir, split_dir, 'images')
            gt_dir = os.path.join(root_dir, split_dir, '1st_manual')
            mask_dir = os.path.join(root_dir, split_dir, 'mask')

        # Collect files
        all_image_paths = sorted(glob.glob(os.path.join(img_dir, '*.tif')))
        all_gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.gif')))
        all_mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.gif')))

        assert len(all_image_paths) > 0, f"No images found in {img_dir}"
        assert len(all_gt_paths) > 0, f"No ground truth found in {gt_dir}"
        assert len(all_image_paths) == len(all_gt_paths), \
            f"Image/GT mismatch: {len(all_image_paths)} vs {len(all_gt_paths)}"

        if not use_test:
            # Fixed split from training set: last 4 as validation (DRIVE has 20 images)
            if val_indices is None:
                val_indices = list(range(16, 20))
            train_indices = [i for i in range(len(all_image_paths)) if i not in val_indices]

            if split == 'train':
                self.image_paths = [all_image_paths[i] for i in train_indices]
                self.gt_paths = [all_gt_paths[i] for i in train_indices]
                self.mask_paths = [all_mask_paths[i] for i in train_indices]
            else:  # val or test (when test has no GT)
                self.image_paths = [all_image_paths[i] for i in val_indices]
                self.gt_paths = [all_gt_paths[i] for i in val_indices]
                self.mask_paths = [all_mask_paths[i] for i in val_indices]
        else:
            self.image_paths = all_image_paths
            self.gt_paths = all_gt_paths
            self.mask_paths = all_mask_paths

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
        """Load mask/gt as binary numpy array [H, W]."""
        mask = np.array(Image.open(path))
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        # Normalize to 0/1
        if mask.max() > 1:
            mask = mask / 255.0
        return (mask > 0.5).astype(np.float32)

    def _resize(self, img, mask, fov_mask):
        """Resize to target size."""
        h, w = self.img_size, self.img_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        fov_mask = cv2.resize(fov_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return img, mask, fov_mask

    def _augment(self, img, mask, fov_mask):
        """Data augmentation."""
        # Horizontal flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
            fov_mask = np.fliplr(fov_mask).copy()

        # Vertical flip
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
            fov_mask = np.flipud(fov_mask).copy()

        # Rotation
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-30, 30)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            fov_mask = cv2.warpAffine(fov_mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        # Scale
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            fov_mask = cv2.resize(fov_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # Crop or pad back to original size
            if new_h > h or new_w > w:
                # Crop center
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                img = img[start_h:start_h + h, start_w:start_w + w]
                mask = mask[start_h:start_h + h, start_w:start_w + w]
                fov_mask = fov_mask[start_h:start_h + h, start_w:start_w + w]
            else:
                # Pad
                pad_h = h - new_h
                pad_w = w - new_w
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')
                fov_mask = np.pad(fov_mask, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Brightness / Contrast
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # contrast
            beta = np.random.uniform(-10, 10)    # brightness
            img = np.clip(alpha * img + beta, 0, 255)

        # Gaussian noise
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 5, img.shape)
            img = np.clip(img + noise, 0, 255)

        return img, mask, fov_mask

    def __getitem__(self, idx):
        # Load
        img = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.gt_paths[idx])
        fov_mask = self._load_mask(self.mask_paths[idx])

        # Resize
        img, mask, fov_mask = self._resize(img, mask, fov_mask)

        # Augment
        if self.augment:
            img, mask, fov_mask = self._augment(img, mask, fov_mask)

        # Normalize to [0, 1]
        img = img / 255.0

        # To tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # [3, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0).float()    # [1, H, W]
        fov_mask = torch.from_numpy(fov_mask).unsqueeze(0).float()  # [1, H, W]

        return {
            'image': img,
            'mask': mask,
            'fov_mask': fov_mask,
            'image_id': os.path.basename(self.image_paths[idx]).split('.')[0],
        }


def get_drive_loaders(
    root_dir,
    img_size=512,
    batch_size=2,
    num_workers=2,
    val_indices=None,
):
    """Create train and validation dataloaders from DRIVE training set."""
    train_dataset = DRIVEDataset(
        root_dir=root_dir,
        split='train',
        img_size=img_size,
        augment=True,
        val_indices=val_indices,
    )
    val_dataset = DRIVEDataset(
        root_dir=root_dir,
        split='val',
        img_size=img_size,
        augment=False,
        val_indices=val_indices,
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
