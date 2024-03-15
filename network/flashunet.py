# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Tuple, Union

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
from .intern_image import InternImageLayer, StemLayer,InternImageBlock
import DCNv3
from .ops_dcnv3 import modules as opsm
class DCN_UNET(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        feature_size: int = 32,
        hidden_size: int = 48,
        depths=[3,3, 6, 6],
        groups=[3,3, 6, 6],
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 2,
        qkv_bias: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.stem = StemLayer(in_chans = in_channels,out_chans = hidden_size)

        self.classification = False
        print('dcn1')
        self.dcn1 = InternImageBlock(core_op = getattr(opsm, 'DCNv3'),channels = hidden_size,groups = groups[0],depth= depths[0])
        print('dcn2')
        self.dcn2 = InternImageBlock(core_op = getattr(opsm, 'DCNv3'),channels = 2*hidden_size,groups = groups[1],depth= depths[1])
        print('dcn2')
        self.dcn3 = InternImageBlock(core_op = getattr(opsm, 'DCNv3'),channels = 4*hidden_size,groups = groups[2],depth= depths[2])

        self.dcn4 = InternImageBlock(core_op = getattr(opsm, 'DCNv3'),channels = 8*hidden_size,groups = groups[3],depth= depths[3],downsample= False)
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size*2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size*2,
            out_channels=feature_size*4,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size*4,
            out_channels=feature_size*8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )


        

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels= 8*hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=4,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        x = self.stem(x_in) #128
        enc1 = self.encoder1(x_in) #512
        
        
        #N, H, W, C = x.shape
        #print(N,H,W,C,'shape')
        
        x1 = self.dcn1(x) #64
        
        enc2 = self.encoder2(x.permute(0,3,1,2)) #128

        x2 = self.dcn2(x1) #32

        enc3 = self.encoder3(x1.permute(0,3,1,2)) # 64
        
        x3 = self.dcn3(x2) #16

        enc4 = self.encoder4(x2.permute(0,3,1,2)) #32

        x4 = self.dcn4(x3) #16
        #print(x4.size(),enc4.size(),'check size')
        dec3 = self.decoder5(x4.permute(0,3,1,2), enc4)
        #print(dec3.size(),enc3.size(),'check')
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        
        return  self.out(out)


def DCN_UNET_2d(spatial_dims=2,in_ch=1,out_ch=2,im_size = 512):

    model = DCN_UNET(
        spatial_dims=spatial_dims,
        img_size = (im_size,im_size),
        in_channels=in_ch,
        out_channels=out_ch,
    )

    return model
