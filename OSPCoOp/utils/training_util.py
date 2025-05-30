import torch
import numpy as np
from PIL import Image
from dassl.data.datasets import build_dataset
from dassl.data.transforms import build_transform
from dassl.data.data_manager import build_data_loader
import random
import torchvision.transforms as transforms
from dassl.data.samplers import build_sampler
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
from dassl.utils import read_image
INTERPOLATION_MODES = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "nearest": Image.NEAREST,
}


class CustomRandomFlip:
    def __init__(self, horizontal=True, vertical=False, p=0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(self, img, mask):
        if self.horizontal and random.random() < self.p:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)

        if self.vertical and random.random() < self.p:
            img = transforms.functional.vflip(img)
            mask = transforms.functional.vflip(mask)

        return img, mask

class CustomRandomResizedCrop:
    def __init__(self, size=(224,224), scale=(0.08, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img,mask):
        area = img.size[0] * img.size[1]
        target_area = torch.randint(int(self.scale[0] * area), int(self.scale[1] * area) + 1, (1,)).item()

        aspect_ratio = 1
        w = int(round((target_area * aspect_ratio) ** 0.5))
        h = int(round((target_area / aspect_ratio) ** 0.5))

        return transforms.functional.resized_crop(img, 0, 0, h, w, self.size),transforms.functional.resized_crop(mask, 0, 0, h, w, self.size)

class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.id_aug_dir = cfg.id_aug_dir
        self.train_root = self.cfg.train_root
        self.data_source = data_source
        self.transform = transform
        self.is_train = is_train
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0
        self.mask_thre_for_resize_crop = cfg.mask_thre_for_resize_crop
        
        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.RandomResizedCrop = CustomRandomResizedCrop()
        self.RandomFlip = CustomRandomFlip()
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        self.to_tensor = T.Compose(to_tensor)
    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        
        if item['domain'] == 1:
            name = item['impath'].split('/')[-1].split('.')[0]
            mask_root = './data/'+self.train_root+'/mask/'+name.split('_')[0]+'_'+name.split('_')[1]+'.png'
            mask_img = Image.open(mask_root)
            mask_img = np.array(mask_img)[:, :, 0]
            item['mask'] = mask_img
        if item['domain'] == 0:
            item['mask'] = np.full((self.cfg.IMAGE_SIZE, self.cfg.IMAGE_SIZE), 255, dtype=np.uint8)
        if item['domain'] == 2:
            item['mask'] = np.full((self.cfg.IMAGE_SIZE, self.cfg.IMAGE_SIZE), 0, dtype=np.uint8)

        img0 = read_image(item['impath'])
        mask = item['mask']

        mask1 = np.stack((mask,) * 3, axis=-1)
        mask1 = Image.fromarray(mask1)
        
        if item['domain'] != 0:
            if self.id_aug_dir:
                img0, mask1 = self.RandomResizedCrop(img0, mask1)
            img1, mask1 = self.RandomFlip(img0, mask1)
        else:
            img1, mask1 = self.RandomFlip(img0, mask1)
        
        mask1 = np.array(mask1)[:, :, 0]

        mask1 = np.where(mask1 > self.cfg.mask_thre, 255, 0)

        img2 = self.transform_img(img1)
        item['img'] = img2
        item['mask'] = mask1
        
        
        if item['domain'] == 1:
            if self.id_aug_dir:
                score = np.sum(mask1 == 255) / np.sum(mask == 255)
                if score < self.mask_thre_for_resize_crop:
                    item['label'] = self.num_classes
                    item['mask'] = np.full((self.cfg.IMAGE_SIZE, self.cfg.IMAGE_SIZE), 0, dtype=np.uint8)
        return item


def build_data_loader(
        cfg,
        sampler_type="SequentialSampler",
        data_source=None,
        batch_size=64,
        n_domain=0,
        n_ins=2,
        tfm=None,
        is_train=True,
        dataset_wrapper=None,
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins,
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )
    assert len(data_loader) > 0

    return data_loader