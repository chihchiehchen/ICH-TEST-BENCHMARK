
from monai import config
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet,UNETR ,SwinUNETR
from monai.transforms import Activations, AddChannel, AsDiscrete, Compose, LoadImage, SaveImage, ScaleIntensity, EnsureType
import torch.nn as nn
from .vanilla_unet import U_Net
from .coordconv import CoordConv2d

def U_Net_vanilla(in_ch =1,out_ch =2,im_size = 512):
    return U_Net(in_ch,out_ch)

def U_Net_mix_kernel(in_ch =1,out_ch =2,im_size = 512):
    return U_Net_mix(in_ch,out_ch)
def Resunet(in_ch=1,out_ch=2,cht_num=16,num_units=2,strides=(1,1,1,1),im_size = 512):

    model = UNet(
        spatial_dims=2,
        in_channels=in_ch,
        out_channels=out_ch,
        channels=(cht_num, cht_num*2, cht_num*4, cht_num*8, cht_num*16),
        strides=strides,
        num_res_units=num_units,
    )

    return model

def UNETR_2d(spatial_dims=2,in_ch=1,out_ch=2,im_size = 512):

    model = UNETR(
        spatial_dims=spatial_dims,
        img_size = (im_size,im_size),
        in_channels=in_ch,
        out_channels=out_ch,
    )

    return model


def SwinUNETR_2d(spatial_dims=2,in_ch=3,out_ch=6,im_size = 512):

    model = SwinUNETR(
        spatial_dims=spatial_dims,
        img_size = (im_size,im_size),
        in_channels=in_ch,
        out_channels=out_ch,
    )

    return model



