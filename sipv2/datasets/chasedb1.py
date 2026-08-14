"""CHASE_DB1 dataset loader."""
import os, glob, numpy as np, torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

class CHASEDB1Dataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=512, augment=True, val_indices=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment
        img_dir = os.path.join(root_dir, 'training', 'images')
        gt_dir = os.path.join(root_dir, 'training', '1st_manual')
        mask_dir = os.path.join(root_dir, 'training', 'mask')
        all_image_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        all_gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
        assert len(all_image_paths) > 0, f"No images in {img_dir}"
        assert len(all_gt_paths) > 0, f"No GT in {gt_dir}"
        if val_indices is None:
            val_indices = list(range(20, 28))
        train_indices = [i for i in range(len(all_image_paths)) if i not in val_indices]
        indices = train_indices if split == 'train' else val_indices
        self.image_paths = [all_image_paths[i] for i in indices]
        self.gt_paths = [all_gt_paths[i] for i in indices]
        self.mask_dir = mask_dir if os.path.isdir(mask_dir) else None

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        img = np.array(Image.open(path))
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        return img.astype(np.float32)

    def _load_mask(self, path):
        mask = np.array(Image.open(path))
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        if mask.max() > 1:
            mask = mask / 255.0
        return (mask > 0.5).astype(np.float32)

    def __getitem__(self, idx):
        img = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.gt_paths[idx])
        if self.mask_dir:
            mask_name = os.path.basename(self.image_paths[idx]).replace('.jpg', '_mask.png')
            mask_path = os.path.join(self.mask_dir, mask_name)
            fov_mask = self._load_mask(mask_path) if os.path.exists(mask_path) else np.ones_like(mask)
        else:
            fov_mask = np.ones_like(mask)
        h, w = self.img_size, self.img_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        fov_mask = cv2.resize(fov_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy(); mask = np.fliplr(mask).copy(); fov_mask = np.fliplr(fov_mask).copy()
            if np.random.rand() > 0.5:
                img = np.flipud(img).copy(); mask = np.flipud(mask).copy(); fov_mask = np.flipud(fov_mask).copy()
        img = img / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        fov_mask = torch.from_numpy(fov_mask).unsqueeze(0).float()
        return {'image': img, 'mask': mask, 'fov_mask': fov_mask,
                'image_id': os.path.basename(self.image_paths[idx]).split('.')[0]}

def get_chasedb1_loaders(root_dir, img_size=512, batch_size=2, num_workers=2, val_indices=None):
    train_dataset = CHASEDB1Dataset(root_dir=root_dir, split='train', img_size=img_size, augment=True, val_indices=val_indices)
    val_dataset = CHASEDB1Dataset(root_dir=root_dir, split='val', img_size=img_size, augment=False, val_indices=val_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
