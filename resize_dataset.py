from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import os, numpy as np
from PIL import Image
from utils.utils import get_subdf
from natsort import natsorted
import PIL,cv2

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop, 
    Blur,   
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    Rotate,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma,
    ShiftScaleRotate     
)

class MNISTResizeDataset(Dataset):
    def __init__(self, img_dir, img_size = 512 , transform=[ShiftScaleRotate(scale_limit=(0,0.20), rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0)]):
        
        self.img_dir = img_dir
        self.mask_dir = img_dir + "_mask"
        self.transform = transform
        self.img_size = img_size
        self.pair_list = []
        self.key_ratio = 255 // 10
        self.prepare_dir()
    def __len__(self):
        return len(self.pair_list)

    def prepare_dir(self):
        for dirPath, dirNames, fileNames in os.walk(self.img_dir):
            for f in fileNames:
                if '.png' in f:
                    img_path = os.path.join(dirPath,f)
                    mask_path = img_path.replace(self.img_dir,self.mask_dir)
                    self.pair_list.append((img_path,mask_path))

        

    def __getitem__(self, idx):
        img_path, mask_path = self.pair_list[idx]
        
        init_image= Image.open(img_path).convert('RGB')
        targetmask = Image.open(mask_path).convert('L')
            
        init_image=init_image.resize((self.img_size, self.img_size))
        targetmask = targetmask.resize((self.img_size, self.img_size),resample =  PIL.Image.NEAREST)  

        init_image = np.array(init_image)
            
        targetmask = np.array(targetmask) #change to numpy
            
        targetmask = np.round((targetmask*(1/self.key_ratio)))
        targetmask = targetmask.astype('int')
        #assert set(np.unique(targetmask).tolist()).issubset(set([x for x in range(self.nb_classes)])) 


        if len(self.transform) > 0:
            aug=Compose(self.transform)
            augmented = aug(image=init_image, mask=targetmask)
            tr=transforms.Compose([transforms.ToTensor() ])
    
            aug_image= tr(augmented["image"]) 
                
            aug_target_mask = augmented['mask']
            return aug_image, aug_target_mask, mask_path
        else:
            image = init_image
            tr = transforms.Compose([   transforms.ToTensor()    ])
            return tr(image), targetmask, mask_path
        
