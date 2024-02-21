# modified from https://github.com/LiheYoung/ST-PlusPlus/blob/master/main.py

from utils import count_params, meanIOU
import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD, AdamW
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.losses import DiceLoss
from network.monai_net import  UNETR_2d,U_Net_vanilla
from network.vanilla_unet import U_Net_coord
from network.archs import UNext,UNext_seg_adapt,UNext_seg_adapt_l

from resize_dataset import MNISTResizeDataset
import torchvision.utils as v_utils
from torch.optim.lr_scheduler import CosineAnnealingLR


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

MODE = None


def parse_args():
    parser = argparse.ArgumentParser(description='MNIST test')

    # basic settings
    
    parser.add_argument('--batch-size', type=int, default=10)
    
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_scheduler', type=bool, default=False)
    parser.add_argument('--lr_min', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--img-key', type=int, default=255)
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--transform', type=str, choices=['resize','center_scaling'],
                        default='resize') 
    parser.add_argument('--model', type=str, choices=['unet','unet_coord','unext_seg_adapt_l','unetr'],
                        default='unet')  #todo 
    parser.add_argument('--opt', type=str, default='AdamW')
    # semi-supervised settings
    
    parser.add_argument('--val-dir', default="/home/chihchiehchen/Exploring-a-Better-Network-Architecture-for-Large-Scale-ICH-Segmentation/mnist_png/mnist_png/testing", type=str, help='val_dir')
    parser.add_argument('--train-dir', default="/home/chihchiehchen/Exploring-a-Better-Network-Architecture-for-Large-Scale-ICH-Segmentation/mnist_png/mnist_png/training", type=str, help='ich_dir')
    parser.add_argument('--in-ch', type=int, default=3)
    parser.add_argument('--nb-classes', type=int, default=11)
    parser.add_argument('--class-weights', type=float, default = [1,1,1,1,1,1,1,1,1,1,1],nargs='+')
    parser.add_argument('--save-model-path', type=str, default='./') #todo

    
    
    
    args = parser.parse_args()
    return args


def main(args):
    
    transform_dict= {'resize':[ShiftScaleRotate(scale_limit=(0,0.20), rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0)],'center_scaling':[ShiftScaleRotate(scale_limit=(-0.8,0.2), rotate_limit=40, shift_limit=0.2, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=0)]}


    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    
    

    valset = MNISTResizeDataset(img_dir = args.val_dir, transform = transform_dict[args.transform])
    valloader = DataLoader(valset, batch_size=4,shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    

    global MODE
    MODE = 'train'

    trainset = MNISTResizeDataset(img_dir = args.train_dir, transform = transform_dict[args.transform])
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=8, drop_last=True)
    
    model, optimizer = init_basic_elems(args)
    
    print('\nParams: %.1fM' % count_params(model))
    
    if not args.lr_scheduler:
        lr_scheduler = None
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max = args.epochs,eta_min = args.lr_min)

    best_model, checkpoints = train(model, trainloader, valloader, optimizer, args,lr_scheduler = lr_scheduler )
    
    

def init_basic_elems(args):
    model_zoo = {'unet': U_Net_vanilla,'unetr': UNETR_2d,'unet_coord':U_Net_coord,'unext_seg_adapt_l':UNext_seg_adapt_l} # todo: setup models
    model = model_zoo[args.model](in_ch =args.in_ch,out_ch = args.nb_classes)

    if args.opt == 'SGD':
        optimizer = SGD([{'params': [param for name, param in model.named_parameters()
                                 ],
                          'lr': args.lr }],
                        lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = AdamW([{'params': [param for name, param in model.named_parameters()
                                 ],
                          'lr': args.lr }],
                        lr=args.lr, weight_decay=1e-5)

    if not args.parallel:
        model = model.cuda() #no dataparrallel
    else:
        model = DataParallel(model).cuda()
    return model, optimizer

def train(model, trainloader, valloader, optimizer, args ,add_normal = None,lr_scheduler = None):
    iters = 0
    total_iters = len(trainloader) * args.epochs

    previous_best = 0.0

    global MODE

    if MODE == 'train':
        checkpoints = []
    if args.nb_classes ==2:
        weights = [1,5] 
        
    else:
        weights = args.class_weights #[1,7,7,7,6,10]
    class_weights = torch.FloatTensor(weights).cuda()
    if add_normal != None:
        dataloader_iterator = iter(add_normal)
    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"] ,previous_best))

        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader)
        
        for i, (element) in enumerate(tbar):
            
            l_img, l_mask,l_mask_name = element[0],element[1],element[2]
            l_img, l_mask = l_img.cuda(), l_mask.cuda()
            
            l_pred = model(l_img)
            
            

            criterion = CrossEntropyLoss(weight = class_weights)
            
            ce_loss = 3*criterion(l_pred, l_mask.long())
            
            criterionD = DiceLoss(include_background=True,to_onehot_y=True, softmax= True,reduction='mean')
            d_loss = criterionD(l_pred, torch.unsqueeze(l_mask,1))
            loss = ce_loss + d_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            iters += 1

            optimizer.param_groups[0]["lr"] = lr
            

            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))
            tbar.set_description('current_loss: %.3f' % (loss))

            if i % 500 == 0 :
                print('ce_loss/d_loss:', ce_loss.item(), d_loss.item())
                
        metric = meanIOU(num_classes=args.nb_classes)

        model.eval()
        tbar = tqdm(valloader)

        with torch.no_grad():
            for i, (element) in enumerate(tbar):
                img, mask = element[0], element[1]
                img = img.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)

                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]

                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

        mIOU *= 100.0
        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_model_path, '%s_%.2f.pth' % (args.model,  previous_best)))
            previous_best = mIOU
            if args.parallel:
                torch.save(model.module.state_dict(),
                       os.path.join(args.save_model_path, '%s_%.2f.pth' % (args.model, mIOU)))
            else:
                torch.save(model.state_dict(),
                       os.path.join(args.save_model_path, '%s_%.2f.pth' % (args.model, mIOU)))

            best_model = deepcopy(model)

        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            if args.parallel:
                torch.save(model.module.state_dict(),
                       os.path.join(args.save_model_path, '%s_%s.pth' % (args.model, str(epoch+1))))
            else:
                torch.save(model.state_dict(),
                       os.path.join(args.save_model_path, '%s_%.2f.pth' % (args.model, mIOU)))
            checkpoints.append(deepcopy(model))
        if lr_scheduler:
            lr_scheduler.step()  
            print(epoch,'lr_scheduler.get_last_lr' ,lr_scheduler.get_last_lr()[0])  
    if MODE == 'train':
        return best_model, checkpoints

    return best_model



if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
