# Exploring-a-Better-Network-Architecture-for-Large-Scale-ICH-Segmentation
This is the official implementation of our work "Exploring a Better Network Architecture for Large-Scale ICH Segmentation". Since our collected ICH datasets are unavailable, we demonstrate our results by using MNIST datasets with the following two settings.

## Results

## MNIST dataset (segmentation, resize to 512 x 512)
We test if network architectures are capatable of detecting long range relations. 

|     Method                   |  Params |  FLOPs  |  mIoU  |
| :-------------------------:  | :-----: | :-----: | :----: |
|     UNet                     |  34.53  |  262.17 |  74.51 |
|     UNet (with  CoordConv)   |  34.53  |  262.70 |  79.00 |
|     UNETR                    |  85.47  |  105.65 |  92.81 |
|     SUNeXt-L (Ours)          |  11.97  |  25.65  |  97.86 |

## MNIST dataset (segmentation, with multi-scale scaling)
We test if network architectures are capatable of capturing multi-scale objects. 
|     Method                   |  Params |  FLOPs  |  mIoU  |
| :-------------------------:  | :-----: | :-----: | :----: |
|     UNet (with  CoordConv)   |  34.53  |  262.70 |  72.12 |
|     UNETR                    |  85.47  |  105.65 |  91.75 |
|     SUNeXt-L (Ours)          |  11.97  |  25.65  |  95.74 |
