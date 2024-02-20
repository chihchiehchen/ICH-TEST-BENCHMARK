# Exploring-a-Better-Network-Architecture-for-Large-Scale-ICH-Segmentation
This is the official implementation of our work "Exploring a Better Network Architecture for Large-Scale ICH Segmentation"

## Results

## MNIST dataset (segmentation, resize to 512 x 512)

|     Method                   |  Params |  FLOPs  |  mIoU  |
| :-------------------------:  | :-----: | :-----: | :----: |
|     UNet                     |  34.53  |  262.17 |  74.51 |
|     UNet (with  CoordConv)   |  34.53  |  262.70 |  78.04 |
|     UNETR                    |  85.47  |  105.65 |  89.92 |
|     SUNeXt-L (Ours)          |  11.97  |  25.65  |  97.55 |
