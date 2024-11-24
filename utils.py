import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
from pathlib import Path
import torch
import logging
from PIL import Image
from matplotlib import pyplot as plt
from slidl.slide import Slide
import pyvips as pv 
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
import copy
from sklearn.metrics import jaccard_score
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from monai.metrics import HausdorffDistanceMetric
import random
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from random import shuffle

def calculate_jaccard_index(preds, y_true):
    # Binarize predictions and labels (assuming preds are probabilities)
    preds_binary = (preds > 0.5).int()  # or use torch.round if preds are logits
    y_true_binary = y_true.int()

    # Remove the channel dimension and flatten
    pred_flat = preds_binary.squeeze(1).view(-1)
    target_flat = y_true_binary.squeeze(1).view(-1)

    # Calculate Jaccard Index
    return jaccard_score(target_flat.cpu().numpy(), pred_flat.cpu().numpy(), average='binary')


class Camel_Train_Dataset(Dataset): 
    def __init__(self, imgs_dir, masks_dir, level0_dir, ids_to_use=None, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.level0_dir = level0_dir

        self.transform = transform
        self.ids = ids_to_use

        # ids_to_use를 기반으로 데이터셋의 각 요소에 대한 파일 경로를 저장합니다.
        
        img_files = [] 
        mask_files = []
        level0_files = [] 
        for idx in ids_to_use:
            img_files1 = glob(os.path.join(imgs_dir, idx, '*', '*'))
            mask_files1 = glob(os.path.join(masks_dir, idx, '*', '*'))
            level0_files1 = glob(os.path.join(level0_dir, idx, '*', '*'))

            img_files.extend(img_files1)
            mask_files.extend(mask_files1)
            level0_files.extend(level0_files1)

        img_files.sort()
        mask_files.sort()
        level0_files.sort()
        
        self.files = [{'image': img, 'mask': mask, 'level0': level0} 
                      for img, mask, level0 in zip(img_files, mask_files, level0_files)]


        print(f"Images: {len(img_files)}, Masks: {len(mask_files)}, Level0: {len(level0_files)}")
        print(f'Creating dataset with {len(self.files)} examples')


    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]

        file = self.transform(file)

        return file

class Camel_Test_Dataset(Dataset):
    def __init__(self, test_dir, test2_dir, ids_to_use=None, transform=None, normal_to_tumor_ratio=100):
        self.test_dir = test_dir
        self.test2_dir = test2_dir
        self.transform = transform
        self.ids = ids_to_use

        # Initialize lists to store file paths
        normal_img_files = []
        tumor_img_files = []
        normal_mask_files = []
        tumor_mask_files = []
        normal_level0_files = []
        tumor_level0_files = []

        # Collect files and classify them
        for idx in ids_to_use:
            img_files = glob(os.path.join(test_dir,'tiles', idx, '*', '*'))
            mask_files = glob(os.path.join(test_dir,'masks', idx, '*', '*'))
            level0_files = glob(os.path.join(test_dir,'tiles_0', idx, '*', '*'))

            img_files.sort()
            mask_files.sort()
            level0_files.sort()

            for img_path, mask_path, level0_path in zip(img_files, mask_files, level0_files):
                if 'normal' in img_path:
                    normal_img_files.append(img_path)
                    normal_mask_files.append(mask_path)
                    normal_level0_files.append(level0_path)
                elif 'tumor' in img_path and not 'non' in img_path:
                    tumor_img_files.append(img_path)
                    tumor_mask_files.append(mask_path)
                    tumor_level0_files.append(level0_path)

        for idx in ids_to_use:
            img_files = glob(os.path.join(test2_dir,'tiles', idx, '*', '*'))
            mask_files = glob(os.path.join(test2_dir,'masks', idx, '*', '*'))
            level0_files = glob(os.path.join(test2_dir,'tiles0', idx, '*', '*'))

            img_files.sort()
            mask_files.sort()
            level0_files.sort()

            for img_path, mask_path, level0_path in zip(img_files, mask_files, level0_files):
                normal_img_files.append(img_path)
                normal_mask_files.append(mask_path)
                normal_level0_files.append(level0_path)

        # Ensure alignment
        normal_img_files.sort()
        tumor_img_files.sort()
        normal_mask_files.sort()
        tumor_mask_files.sort()
        normal_level0_files.sort()
        tumor_level0_files.sort()

        # Calculate ratio
        tumor_count = len(tumor_img_files)
        normal_count = int(tumor_count * (normal_to_tumor_ratio / 100))
        if normal_count > len(normal_img_files):
            normal_count = len(normal_img_files)  # Cap the normal_count to avoid out-of-range errors
        print(f'tumor_count:{tumor_count} and normal_count: {normal_count}')
        # Select indices according to the ratio
        tumor_indices = np.random.choice(len(tumor_img_files), tumor_count, replace=False)
        normal_indices = np.random.choice(len(normal_img_files), normal_count, replace=False)

        # Combine and shuffle
        balanced_files = [{'image': normal_img_files[i], 'mask': normal_mask_files[i], 'level0': normal_level0_files[i],'id': self.extract_id(normal_img_files[i]),'name':normal_mask_files[i]}
                          for i in normal_indices] + \
                         [{'image': tumor_img_files[i], 'mask': tumor_mask_files[i], 'level0': tumor_level0_files[i],'id': self.extract_id(tumor_img_files[i]),'name':tumor_mask_files[i]}
                          for i in tumor_indices]
        shuffle(balanced_files)

        self.files = balanced_files
        print(f'Creating balanced dataset with {len(self.files)} examples (normal_to_tumor_ratio: {normal_to_tumor_ratio}%)')

    def extract_id(self, filename):
        # 파일명에서 ID 추출
        return filename.split('/')[-1].split('_')[1]
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        """Fetch a data sample for a given index."""
        file = self.files[i]
        if self.transform:
            file = self.transform(file)
        return file
class PAIP_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, normal_to_tumor_ratio=100):
        self.root_dir = root_dir
        self.transform = transform
        self.files = self.load_files(normal_to_tumor_ratio)

    def extract_id(self, filename):
        # 파일명에서 ID 추출
        return filename.split('/')[-1].split('_')[1]

    def is_mask_empty(self, mask_path):
        # 마스크 이미지를 로드하고 전부 0인지 확인
        mask = np.array(Image.open(mask_path))
        return np.all(mask == 0)

    def load_files(self, ratio):
        img_files = sorted(glob(os.path.join(self.root_dir, 'patch1', '*.png')))
        mask_files = sorted(glob(os.path.join(self.root_dir, 'mask', '*.png')))
        level0_files = sorted(glob(os.path.join(self.root_dir, 'patch0', '*.png')))

        normal_img_files = []
        normal_mask_files = []
        normal_level0_files = []
        tumor_img_files = []
        tumor_mask_files = []
        tumor_level0_files = []

        for img, mask, level0 in zip(img_files, mask_files, level0_files):
            if self.is_mask_empty(mask):
                normal_img_files.append(img)
                normal_mask_files.append(mask)
                normal_level0_files.append(level0)
            else:
                tumor_img_files.append(img)
                tumor_mask_files.append(mask)
                tumor_level0_files.append(level0)

        tumor_count = len(tumor_img_files)
        normal_count = int(tumor_count * (ratio / 100))
        normal_count = min(normal_count, len(normal_img_files))
        print(f'normal_count = {normal_count} , tumor_count= {tumor_count}')
        tumor_indices = np.random.choice(len(tumor_img_files), tumor_count, replace=False)
        normal_indices = np.random.choice(len(normal_img_files), normal_count, replace=False)

        balanced_files = [{'image': normal_img_files[i], 'mask': normal_mask_files[i], 'level0': normal_level0_files[i], 'id': self.extract_id(normal_img_files[i])}
                          for i in normal_indices] + \
                         [{'image': tumor_img_files[i], 'mask': tumor_mask_files[i], 'level0': tumor_level0_files[i], 'id': self.extract_id(tumor_img_files[i])}
                          for i in tumor_indices]
        shuffle(balanced_files)

        print(f'Creating balanced dataset with {len(balanced_files)} examples (normal_to_tumor_ratio: {ratio}%)')
        return balanced_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        if self.transform:
            file = self.transform(file)
        return file

        
class PAIP_Dataset2(Dataset):
    def __init__(self, root_dir, transform=None, normal_to_tumor_ratio=100):
        self.root_dir = root_dir
        self.transform = transform
        self.files = self.load_files(normal_to_tumor_ratio)

    def extract_id(self, filename):
        # 파일명에서 ID 추출
        return filename.split('/')[-1].split('_')[1]

    def is_mask_empty(self, mask_path):
        # 마스크 이미지를 로드하고 전부 0인지 확인
        mask = np.array(Image.open(mask_path))
        return np.all(mask == 0)

    def load_files(self, ratio):
        img_files = sorted(glob(os.path.join(self.root_dir, 'patch1', '*.png')))
        mask_files = sorted(glob(os.path.join(self.root_dir, 'mask', '*.png')))
        level0_files = sorted(glob(os.path.join(self.root_dir, 'patch0', '*.png')))

        normal_img_files = []
        normal_mask_files = []
        normal_level0_files = []
        tumor_img_files = []
        tumor_mask_files = []
        tumor_level0_files = []

        for img, mask, level0 in zip(img_files, mask_files, level0_files):
            if self.is_mask_empty(mask):
                normal_img_files.append(img)
                normal_mask_files.append(mask)
                normal_level0_files.append(level0)
            else:
                tumor_img_files.append(img)
                tumor_mask_files.append(mask)
                tumor_level0_files.append(level0)
        normal_count = int(len(normal_img_files)*4//5)

        tumor_count = int(len(normal_img_files) * (100 / ratio))
        tumor_count = min(tumor_count, len(tumor_img_files))
        print(f'normal_count = {normal_count} , tumor_count= {tumor_count}')
        np.random.seed(421)
        tumor_indices = np.random.choice(len(tumor_img_files), tumor_count, replace=False)
        normal_indices = np.random.choice(len(normal_img_files), normal_count, replace=False)

        balanced_files = [{'image': normal_img_files[i], 'mask': normal_mask_files[i], 'level0': normal_level0_files[i], 'id': self.extract_id(normal_img_files[i])}
                          for i in normal_indices] + \
                         [{'image': tumor_img_files[i], 'mask': tumor_mask_files[i], 'level0': tumor_level0_files[i], 'id': self.extract_id(tumor_img_files[i])}
                          for i in tumor_indices]
        shuffle(balanced_files)

        print(f'Creating balanced dataset with {len(balanced_files)} examples (normal_to_tumor_ratio: {ratio}%)')
        return balanced_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        if self.transform:
            file = self.transform(file)
        return file

class PAIP_Dataset3(Dataset):
    def __init__(self, root_dir, patient_id, transform=None, normal_to_tumor_ratio=100):
        self.root_dir = root_dir
        self.patient_id = patient_id
        self.transform = transform
        self.files = self.load_files(normal_to_tumor_ratio)

    def extract_id(self, filename):
        # Extract patient ID from the filename
        return filename.split('/')[-1].split('_')[1]

    def extract_x(self, filename):
        return filename.split('/')[-1].split('_')[2]
    
    def extract_y(self, filename):
        return filename.split('/')[-1].split('_')[3]

    def is_mask_empty(self, mask_path):
        # Check if the mask is empty
        mask = np.array(Image.open(mask_path))
        return np.all(mask == 0)

    def load_files(self, ratio):
        img_files = sorted(glob(os.path.join(self.root_dir, 'patch1', f'*_{self.patient_id}_*.png')))
        mask_files = sorted(glob(os.path.join(self.root_dir, 'mask', f'*_{self.patient_id}_*.png')))
        level0_files = sorted(glob(os.path.join(self.root_dir, 'patch0', f'*_{self.patient_id}_*.png')))

        normal_img_files = []
        normal_mask_files = []
        normal_level0_files = []
        tumor_img_files = []
        tumor_mask_files = []
        tumor_level0_files = []

        for img, mask, level0 in zip(img_files, mask_files, level0_files):
            if self.is_mask_empty(mask):
                normal_img_files.append(img)
                normal_mask_files.append(mask)
                normal_level0_files.append(level0)
            else:
                tumor_img_files.append(img)
                tumor_mask_files.append(mask)
                tumor_level0_files.append(level0)

        normal_count = int(len(normal_img_files) * 4 // 5)
        tumor_count = int(len(normal_img_files) * (100 / ratio))
        tumor_count = min(tumor_count, len(tumor_img_files))
        np.random.seed(421)
        tumor_indices = np.random.choice(len(tumor_img_files), tumor_count, replace=False)
        normal_indices = np.random.choice(len(normal_img_files), normal_count, replace=False)

        balanced_files = [{'image': normal_img_files[i], 'mask': normal_mask_files[i], 'level0': normal_level0_files[i], 'id': self.extract_id(normal_img_files[i]), 'x': self.extract_x(normal_img_files[i]) , 'y': self.extract_y(normal_img_files[i])}
                          for i in normal_indices] + \
                         [{'image': tumor_img_files[i], 'mask': tumor_mask_files[i], 'level0': tumor_level0_files[i], 'id': self.extract_id(tumor_img_files[i]), 'x': self.extract_x(tumor_img_files[i]) , 'y': self.extract_y(tumor_img_files[i])}
                          for i in tumor_indices]
        shuffle(balanced_files)

        print(f'Creating balanced dataset with {len(balanced_files)} examples (normal_to_tumor_ratio: {ratio}%)')
        return balanced_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        if self.transform:
            file = self.transform(file)
        return file


class CMC_Dataset0(Dataset):
    def __init__(self, root_dir, transform=None, ratio=100):
        self.root_dir = root_dir
        self.transform = transform
        self.files = self.load_files(ratio)

    def extract_id(self, filename):
        # 파일명에서 ID 추출
        return filename.split('/')[-1].split('_')[1]

    def is_mask_empty(self, mask_path):
        # 마스크 이미지를 로드하고 전부 0인지 확인
        mask = np.array(Image.open(mask_path))
        return np.all(mask == 0)

    def load_files(self, ratio):
        normal_img_files = sorted(glob(os.path.join(self.root_dir,'normal','*', 'patch2', '*.png')))
        normal_mask_files = sorted(glob(os.path.join(self.root_dir,'normal','*', 'mask', '*.png')))
        normal_level0_files = sorted(glob(os.path.join(self.root_dir,'normal','*', 'patch0', '*.png')))

        tumor_img_files = sorted(glob(os.path.join(self.root_dir,'tumor','*', 'patch2', '*.png')))
        tumor_mask_files = sorted(glob(os.path.join(self.root_dir,'tumor','*', 'mask', '*.png')))
        tumor_level0_files = sorted(glob(os.path.join(self.root_dir,'tumor','*', 'patch0', '*.png')))


        tumor_count = len(tumor_img_files)
        normal_count = int(tumor_count * (ratio / 100))
        normal_count = min(normal_count, len(normal_img_files))


        print(f'normal_count = {normal_count} , tumor_count= {tumor_count}')
        tumor_indices = np.random.choice(len(tumor_img_files), tumor_count, replace=False)
        normal_indices = np.random.choice(len(normal_img_files), normal_count, replace=False)

        balanced_files = [{'image': normal_img_files[i], 'mask': normal_mask_files[i], 'level0': normal_level0_files[i], 'id': self.extract_id(normal_img_files[i])}
                          for i in normal_indices] + \
                         [{'image': tumor_img_files[i], 'mask': tumor_mask_files[i], 'level0': tumor_level0_files[i], 'id': self.extract_id(tumor_img_files[i])}
                          for i in tumor_indices]
        shuffle(balanced_files)

        print(f'Creating balanced dataset with {len(balanced_files)} examples (normal_to_tumor_ratio: {ratio}%)')
        return balanced_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        if self.transform:
            file = self.transform(file)
        return file

class CMC_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, ratio=100):
        self.root_dir = root_dir
        self.transform = transform
        self.files = self.load_files(ratio)

    def extract_id(self, filename):
        # 파일명에서 ID 추출
        return filename.split('/')[-1].split('_')[1]

    def is_mask_empty(self, mask_path):
        # 마스크 이미지를 로드하고 전부 0인지 확인
        mask = np.array(Image.open(mask_path))
        return np.all(mask == 0)

    def load_files(self, ratio):
        # Initialize lists to store file paths
        normal_img_files = sorted(glob(os.path.join(self.root_dir,'normal','*', 'patch2', '*.png')))
        normal_mask_files = sorted(glob(os.path.join(self.root_dir,'normal','*', 'mask', '*.png')))
        normal_level0_files = sorted(glob(os.path.join(self.root_dir,'normal','*', 'patch0', '*.png')))

        tumor_img_files = sorted(glob(os.path.join(self.root_dir,'tumor','*', 'patch2', '*.png')))
        tumor_mask_files = sorted(glob(os.path.join(self.root_dir,'tumor','*', 'mask', '*.png')))
        tumor_level0_files = sorted(glob(os.path.join(self.root_dir,'tumor','*', 'patch0', '*.png')))
  

        tumor_count = len(tumor_img_files)
        normal_count = int(tumor_count * (ratio / 100))
        normal_count = min(normal_count, len(normal_img_files))


        print(f'normal_count = {normal_count} , tumor_count= {tumor_count}')
        tumor_indices = np.random.choice(len(tumor_img_files), tumor_count, replace=False)
        normal_indices = np.random.choice(len(normal_img_files), normal_count, replace=False)

        balanced_files = [{'image': normal_img_files[i], 'mask': normal_mask_files[i], 'level0': normal_level0_files[i], 'id': self.extract_id(normal_img_files[i])}
                          for i in normal_indices] + \
                         [{'image': tumor_img_files[i], 'mask': tumor_mask_files[i], 'level0': tumor_level0_files[i], 'id': self.extract_id(tumor_img_files[i])}
                          for i in tumor_indices]
        shuffle(balanced_files)

        print(f'Creating balanced dataset with {len(balanced_files)} examples (normal_to_tumor_ratio: {ratio}%)')
        return balanced_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        if self.transform:
            file = self.transform(file)
        return file





def seed_everything(seed=42):
    random.seed(seed) # Python 내장 random 모듈의 시드 설정
    os.environ['PYTHONHASHSEED'] = str(seed) # 환경변수를 통한 Python 해시 시드 설정
    np.random.seed(seed) # NumPy의 시드 설정
    torch.manual_seed(seed) # PyTorch의 시드 설정
    torch.cuda.manual_seed(seed) # PyTorch의 CUDA 시드 설정
    torch.cuda.manual_seed_all(seed) # 멀티 GPU를 위한 PyTorch의 CUDA 시드 설정
    torch.backends.cudnn.deterministic = True # cudnn의 알고리즘 결정성 활성화
    torch.backends.cudnn.benchmark = False # False로 설정 시, 네트워크의 입력 데이터 크기가 변하지 않을 때 최적의 알고리즘을 찾는 과정이 비활성화됩니다.
    

def visualizeSegmentationAugmentation(path_to_image, path_to_mask, transform, folder=None):
    img = Image.open(path_to_image)
    mask = Image.open(path_to_mask)
    img_nd = np.array(img)
    mask_nd = np.array(mask)

    #if transform is not None:
    transformed = transform(image=img_nd, mask=mask_nd)

    fontsize = 18

    f, ax = plt.subplots(2, 2, figsize=(8, 8))

    ax[0, 0].imshow(img_nd)
    ax[0, 0].set_title('Original image', fontsize=fontsize)

    ax[1, 0].imshow(mask_nd)
    ax[1, 0].set_title('Original mask', fontsize=fontsize)

    ax[0, 1].imshow(transformed['image'])
    ax[0, 1].set_title('Transformed image', fontsize=fontsize)

    ax[1, 1].imshow(transformed['mask'])
    ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

    if folder:
        plt.savefig(os.path.join(folder, Path(path_to_image).stem+'_augmentationexample.jpg'))


