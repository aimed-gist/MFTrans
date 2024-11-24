from torch.utils.data import DataLoader
import pytorch_lightning as pl
from PIL import Image
from wsi_utils import *
import torch
from monai.transforms import *
from utils import PAIP_Dataset,PAIP_Dataset2,CMC_Dataset, Camel_Train_Dataset,Camel_Test_Dataset 
from monai.data.image_reader import PILReader
from torchvision.transforms import ColorJitter
class RandColorAdjust(Transform):
    def __init__(self, keys, prob=0.5, hue=0.2, saturation=0.2):
        self.keys = keys
        self.prob = prob
        self.hue = hue
        self.saturation = saturation
        self.color_jitter = ColorJitter(hue=self.hue, saturation=self.saturation)
        self.apply_transform = torch.rand(1).item() < self.prob

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if self.apply_transform:
                d[key] = self.color_jitter(d[key])
        return d

class Level0ToImageTransform(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            file_path = d[key]
            level0_data = np.load(file_path, allow_pickle=True)
            tiles_grid = level0_data.reshape(4, 4, level0_data.shape[1], level0_data.shape[2], level0_data.shape[3])
            rows = [np.vstack(tiles_grid[i, :, :, :, :3]) for i in range(4)]
            final_image = np.hstack(rows)
            final_image = np.transpose(final_image, (2, 0, 1))  # CHW 형식으로 변환
            d[key] = final_image
        return d

class Level0ToImage(MapTransform):
    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            file_path = d[key]
            file_extension = os.path.splitext(file_path)[1]
            if file_extension == '.npy':
                level0_data = np.load(d[key], allow_pickle=True)
                level0_data = np.transpose(level0_data, (2, 1, 0))
                d[key] = level0_data[:3, :, :]
            else:
                with Image.open(file_path) as img:
                    final_image = np.array(img)
                    final_image = np.transpose(final_image, (2, 1, 0))
                    d[key] = final_image
        return d

class ChannelCut(MapTransform):
    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = d[key][:3, :, :]
        return d

def split_files_by_id(files, ids, train_ratio, val_ratio, test_ratio):
    train_ids, temp_ids = train_test_split(ids, train_size=train_ratio, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
    train_files = [file for file in files if file['id'] in train_ids]
    val_files = [file for file in files if file['id'] in val_ids]
    test_files = [file for file in files if file['id'] in test_ids]
    return train_files, val_files, test_files

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, fold=4, batch_size=32, resize=False, data='camelyon', ratio=100):
        super().__init__()
        self.ratio = ratio
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.fold = fold
        self.data = data

        if self.data == 'camelyon':
            self._init_camelyon_transforms(resize)
        elif self.data == 'paip':
            self._init_paip_transforms(resize)
        else:
            self._init_cmc_transforms(resize)

    def _init_camelyon_transforms(self, resize):
        if resize:
            resize_transform = Resized(keys=["image"], spatial_size=resize, mode='bilinear')
            resize_transform2 = Resized(keys=["mask"], spatial_size=resize, mode='nearest')
            self.train_transform = Compose([
                LoadImaged(keys=["image", "mask"], ensure_channel_first=True, reader=PILReader),
                RandFlipd(keys=["image", "mask"], prob=0.2),
                RandRotate90d(keys=["image", "mask"], prob=0.2, max_k=3),
                NormalizeIntensityd(keys=["image", "level0"], subtrahend=[194, 139, 178], divisor=[41, 50.5, 37], nonzero=False, channel_wise=True),
                RandColorAdjust(keys=["image"], prob=0.2, hue=0.3, saturation=0.1),
                resize_transform,
                resize_transform2,
                ToTensord(keys=["image", "mask"], dtype=torch.float32)
            ])
            self.val_transform = self.train_transform
            self.test_transform = self.train_transform
        else:
            self.train_transform = Compose([
                LoadImaged(keys=["image", "mask"], ensure_channel_first=True, reader=PILReader),
                Level0ToImageTransform(keys=['level0']),
                RandFlipd(keys=["image", "mask", 'level0'], prob=0.2, spatial_axis=1),
                RandRotate90d(keys=["image", "mask", 'level0'], prob=0.2, max_k=3),
                NormalizeIntensityd(keys=["image", "level0"], subtrahend=[194, 139, 178], divisor=[41, 50.5, 37], nonzero=False, channel_wise=True),
                RandColorAdjust(keys=["image", 'level0'], prob=0.2, hue=0.3, saturation=0.1),
                ToTensord(keys=["image", "mask", "level0"], dtype=torch.float32)
            ])
            self.val_transform = self.train_transform
            self.test_transform = Compose([
                LoadImaged(keys=["image", "mask"], ensure_channel_first=True, reader=PILReader),
                Level0ToImage(keys=["level0"]),
                ToTensord(keys=["image", "mask", "level0"], dtype=torch.float32)
            ])

    def _init_paip_transforms(self, resize):
        resize_transform = Resized(keys=["image"], spatial_size=resize, mode='bilinear')
        resize_transform2 = Resized(keys=["mask"], spatial_size=resize, mode='nearest')
        self.train_transform = Compose([
            LoadImaged(keys=["image", "mask", "level0"], ensure_channel_first=True, reader=PILReader),
            ChannelCut(keys=["image", "level0"]),
            RandFlipd(keys=["image", "mask", 'level0'], prob=0.2, spatial_axis=1),
            RandRotate90d(keys=["image", "mask", 'level0'], prob=0.2, max_k=3),
            NormalizeIntensityd(keys=["image", "level0"], subtrahend=[184, 149, 195], divisor=[30, 43, 31], nonzero=False, channel_wise=True),
            resize_transform,
            resize_transform2,
            ToTensord(keys=["image", "mask", "level0"], dtype=torch.float32)
        ])
        self.val_transform = self.train_transform
        self.test_transform = self.train_transform

    def _init_cmc_transforms(self, resize):
        resize_transform = Resized(keys=["image"], spatial_size=resize, mode='bilinear')
        resize_transform2 = Resized(keys=["mask"], spatial_size=resize, mode='nearest')
        self.train_transform = Compose([
            LoadImaged(keys=["image", "mask", "level0"], ensure_channel_first=True, reader=PILReader),
            ChannelCut(keys=["image", "level0"]),
            RandFlipd(keys=["image", "mask", 'level0'], prob=0.2, spatial_axis=1),
            RandRotate90d(keys=["image", "mask", 'level0'], prob=0.2, max_k=3),
            NormalizeIntensityd(keys=["image", "level0"], subtrahend=[194, 139, 178], divisor=[41, 50.5, 37], nonzero=False, channel_wise=True),
            resize_transform,
            resize_transform2,
            ToTensord(keys=["image", "mask", "level0"], dtype=torch.float32)
        ])
        self.val_transform = self.train_transform
        self.test_transform = self.train_transform

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.data == 'camelyon':
                self._setup_camelyon()
            elif self.data == 'paip':
                self._setup_paip()
            else:
                self._setup_cmc()
        if stage == 'test' or stage is None:
            if self.data == 'camelyon':
                self._setup_camelyon_test()
            elif self.data == 'paip':
                self._setup_paip_test()
            else:
                self._setup_cmc_test()

    def _setup_camelyon(self):
        analysis_dir_path = os.path.join(self.data_dir, 'camelyon16')
        dir_img_train = os.path.join(analysis_dir_path, 'train', 'tiles')
        dir_mask_train = os.path.join(analysis_dir_path, 'train', 'masks')
        dir_level0_train = os.path.join(analysis_dir_path, 'train', 'tiles_0')
        train_cases = [name for name in os.listdir(dir_level0_train) if os.path.isdir(os.path.join(dir_level0_train, name))]
        train_ids = np.array(train_cases)
        np.random.shuffle(train_ids)
        k_fold = 5
        fold_size = len(train_ids) // k_fold
        train_ids_folds = [train_ids[i:i + fold_size] for i in range(0, len(train_ids), fold_size)]
        val_ids_fold = train_ids_folds[self.fold]
        train_ids_fold = np.concatenate([train_ids_folds[i] for i in range(k_fold) if i != self.fold])
        self.train_dataset = Camel_Train_Dataset(imgs_dir=dir_img_train, masks_dir=dir_mask_train, level0_dir=dir_level0_train,
                                                         ids_to_use=train_ids_fold, transform=self.train_transform)
        self.val_dataset = Camel_Train_Dataset(imgs_dir=dir_img_train, masks_dir=dir_mask_train, level0_dir=dir_level0_train,
                                                       ids_to_use=val_ids_fold, transform=self.val_transform)

    def _setup_paip(self):
        analysis_dir_path = os.path.join(self.data_dir, 'paip2019')
        train_dir = os.path.join(analysis_dir_path, 'train')
        val_dir = os.path.join(analysis_dir_path, 'val')
        self.train_dataset = PAIP_Dataset(train_dir, transform=self.train_transform, normal_to_tumor_ratio=self.ratio)
        self.val_dataset = PAIP_Dataset(val_dir, transform=self.val_transform, normal_to_tumor_ratio=self.ratio)

    def _setup_cmc(self):
        analysis_dir_path = os.path.join(self.data_dir, 'cmc')
        train_dir = os.path.join(analysis_dir_path, 'train')
        val_dir = os.path.join(analysis_dir_path, 'val')
        self.train_dataset = CMC_Dataset(train_dir, transform=self.train_transform, ratio=self.ratio)
        self.val_dataset = CMC_Dataset(val_dir, transform=self.val_transform, ratio=self.ratio)

    def _setup_camelyon_test(self):
        analysis_dir_path = os.path.join(self.data_dir, 'camelyon16')
        test_dir = os.path.join(analysis_dir_path, 'test')
        test2_dir = os.path.join(analysis_dir_path, 'test3')
        dir_level0_test = os.path.join(analysis_dir_path, 'test', 'tiles_0')
        test_cases = [name for name in os.listdir(dir_level0_test) if os.path.isdir(os.path.join(dir_level0_test, name))]
        self.test_dataset = Camel_Test_Dataset(test_dir, test2_dir, normal_to_tumor_ratio=self.ratio,
                                                        ids_to_use=test_cases, transform=self.test_transform)

    def _setup_paip_test(self):
        test_dir = os.path.join(self.data_dir, 'paip2019','test')
        self.test_dataset = PAIP_Dataset2(test_dir, transform=self.test_transform, normal_to_tumor_ratio=self.ratio)

    def _setup_cmc_test(self):
        test_dir = os.path.join(self.data_dir, 'cmc','test')
        self.test_dataset = CMC_Dataset(test_dir, transform=self.test_transform, ratio=self.ratio)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=12, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False, pin_memory=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False, pin_memory=False, drop_last=True)