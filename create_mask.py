import argparse

import os, shutil
import numpy as np 
from PIL import Image 

def parse_args():
    
    parser.add_argument('--dir-list', type=str, default = ["/home/chihchiehchen/Exploring-a-Better-Network-Architecture-for-Large-Scale-ICH-Segmentation/mnist_png/mnist_png/training","/home/chihchiehchen/Exploring-a-Better-Network-Architecture-for-Large-Scale-ICH-Segmentation/mnist_png/mnist_png/testing"],nargs='+')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    key_ratio = 255 // 10
    for dirname in args.dir_list:
        mask_dir = dirname + "_mask"
    

        for dir_label in os.listdir(dirname):
            l_int = int(dir_label) + 1
            sub_dir = os.path.join(dirname,dir_label)
            sub_mask_dir = os.path.join(mask_dir,dir_label)
            os.makedirs(sub_mask_dir, exist_ok = True)
            for f in os.listdir(sub_dir):
                png_name = os.path.join(sub_dir,f)
                f_name = os.path.join(sub_mask_dir,f)
            
                assert not os.path.exists(f_name)
                img = np.array(Image.open(png_name).convert('L'))
                m = np.max(img)
                mask = np.where(img >= m/2,l_int*key_ratio, 0 )

                final_img = Image.fromarray(np.uint8(mask))
                final_img.save(f_name)
            
