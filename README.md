# Exploring-a-Better-Network-Architecture-for-Large-Scale-ICH-Segmentation
This is the official implementation of our work "Exploring a Better Network Architecture for Large-Scale ICH Segmentation". Since our collected ICH datasets are unavailable, we demonstrate our results by using MNIST datasets with the following two settings.

## Create masks for MNIST simulated segmentation dataset
   Extract mnist_png.tar.gz in the repository, find train and test directories and run 
   ```bash
   python create_mask.py --dir_list address_of_training_directories address_of_testing_directories
   ```

## Requirements
   ```bash
   pip install torch torchvision albumentations  monai==1.2.0 timm==0.9.12
   ```


## Results

## MNIST dataset (segmentation, resize to 512 x 512)
   We test if network architectures are capatable of detecting long range relations. Run the following command:
    
   ```bash
   python main_train.py --model unet(unet_coord/unetr/unext_seg_adapt_l/swin_unetr)
   ```
    
   |     Method                   |  Params |  FLOPs  |  mIoU  |
   | :-------------------------:  | :-----: | :-----: | :----: |
   |     UNet                     |  34.53  |  262.17 |  74.51 |
   |     UNet (with  CoordConv)   |  34.53  |  262.70 |  79.00 |
   |     UNETR                    |  85.47  |  105.65 |  93.59 |
   |     SUNeXt-L (Ours)          |  11.97  |  25.65  |  97.66 |

## MNIST dataset (segmentation, with multi-scale scaling)
    We test if network architectures are capatable of capturing multi-scale objects. 


   |     Method                   |  Params |  FLOPs  |  mIoU  |
   | :-------------------------:  | :-----: | :-----: | :----: |
   |     UNETR                    |  85.47  |  105.65 |  73.13 |
   |     SwinUNETR                |   6.28  |   19.61 |  92.69 |
   |     SUNeXt-L (Ours)          |  11.97  |  25.65  |  93.72 |
