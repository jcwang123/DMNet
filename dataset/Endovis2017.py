import os
import glob
import cv2
import json
import math
import time
import numpy as np
import sys,random
from PIL import Image
sys.path.append('/raid/wjc/code/RealtimeSegmentation/')

import torch
import torch.utils.data as data
import torch.nn.functional as F

# from utils.image import get_border, get_affine_transform, affine_transform, color_aug

import torchvision.transforms.functional as TF
from torchvision import transforms
MEAN = [0.40789654, 0.44719302, 0.47026115]
STD = [0.28863828, 0.27408164, 0.27809835]

# Folds = [[1,3],[2,6],[4,8],[5,7]]
Folds = [[1,3],[2,5],[4,8],[6,7]]
ins_types = ['Bipolar_Forceps','Prograsp_Forcep','Large_Needle_Driver','Vessel_Sealer', 'Grasping_Retractor',
            'Monopolar_Curved_Scissors','Other']


class endovis2017(data.Dataset):
    def __init__(self, split, t=1, fold=0, rate=4, tag='part', global_n=0, test=False):
        super(endovis2017, self).__init__()
        self.split = split
        self.mean = np.array(MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(STD, dtype=np.float32)[None, None, :]
        self.img_size = {'h': 512, 'w': 640}
        self.t = t
        self.tag = tag
        self.test = test
        self.class_num = 7 if tag=='type' else 4
        # /raid/wjc/data/ead/endovis2017/training/instrument_dataset_*/frame.png
        # 8 * 225 frames
        # 0 for valid and the last 25 of the rest are used for valid
        
        if test:
            self.images = [[j,i] for j in range(1,9) for i in range(225,300)] + [[j,i] for j in range(9,11) for i in range(300)]
        else:
            self.images = []
            train_images = []
            valid_images = []
            for f in range(4):
                if f==fold:
                    valid_images += [[j,i] for j in Folds[f] for i in range(225)]
                else:
                    train_images += [[j,i] for j in Folds[f] for i in range(225)]
            self.images = train_images if self.split=='train' else valid_images         
        print('Loaded {}frames'.format(len(self.images)))
        self.num_samples = len(self.images)
        self.rate = rate
        self.global_n = global_n
        
        
    def load_data(self, ins,frame,t=1,global_n=0):
        image = [] 
        if global_n:
            global_images_index = (np.random.rand(global_n)*225).astype('int')
            image += [np.load('/raid/wjc/data/ead/endovis2017/training/instrument_dataset_{}/processed_v1/image{:03d}.npy'\
                                  .format(ins,ind)) for ind in global_images_index]
        if t>frame:
            image += list([np.load('/raid/wjc/data/ead/endovis2017/training/instrument_dataset_{}/processed_v1/image{:03d}.npy'\
                                  .format(ins,i)) for i in range(frame+t-1,frame-1,-1)])
        else:
            image += list([np.load('/raid/wjc/data/ead/endovis2017/training/instrument_dataset_{}/processed_v1/image{:03d}.npy'\
                                  .format(ins,i)) for i in range(frame-t+1,frame+1)])
        label = np.load('/raid/wjc/data/ead/endovis2017/training/instrument_dataset_{}/processed_v1/{}{:03d}.npy'.format(ins,self.tag,frame))
        return image, label
    
    def transform(self, images, masks):
        # Resize
        scale = random.random()*0.4+1
        resize = transforms.Resize(size=(int(512*scale), int(640*scale)))
        
        
#         image = resize(image)
#         mask = resize(mask)
        images = list([resize(image) for image in images])
        masks = list([resize(mask) for mask in masks])
        

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            images[0], output_size=(512, 640))
#         image = TF.crop(image, i, j, h, w)
#         mask = resize(mask)
        images = list([TF.crop(image, i, j, h, w) for image in images])
        masks = list([TF.crop(mask, i, j, h, w) for mask in masks])
        
        # Random horizontal flipping
        if random.random() > 0.5:
#             image = TF.hflip(image)
            images = list([TF.hflip(image) for image in images])
            masks = list([TF.hflip(mask) for mask in masks])

        # Random vertical flipping
        if random.random() > 0.5:
#             image = TF.vflip(image)
            images = list([TF.vflip(image) for image in images])
            masks = list([TF.vflip(mask) for mask in masks])

        return images, masks
    
    def __getitem__(self, index):
        
        ins,frame = self.images[index]
#         print(img_path)
#         st = time.perf_counter()
        imgs, label = self.load_data(ins, frame, self.t, global_n=self.global_n)
#         print('Load data:',time.perf_counter()-st)
#         st = time.perf_counter()
        
        label = (label/30+0.5).astype('int') # w * h
        masks = []
        
        if self.split=='train':
#             img = Image.fromarray(np.uint8(img))
            imgs = [Image.fromarray(np.uint8(img)) for img in imgs]
            classes = np.unique(label)
            masks = []
            for cls in classes:
                if cls:
                    masks.append(Image.fromarray(np.uint8(label==cls)))
            imgs,masks = self.transform(imgs,masks)
            imgs = [np.asarray(img) for img in imgs]
            label = np.zeros((imgs[0].shape[0],imgs[0].shape[1]))
            for i in range(1, len(classes)):
                mask = np.asarray(masks[i-1])
                label[mask>0] = classes[i]
#             print('transform data:',time.perf_counter()-st)
#             st = time.perf_counter()    
        imgs = np.array(imgs)
#         print('img2numpy:',time.perf_counter()-st)
#         st = time.perf_counter()    
        img2 = imgs - np.min(imgs)
        img2 = img2 / np.max(img2)
#         img2 = imgs/255
        img2 = (img2 - self.mean) / self.std
#         print('imgmean:',time.perf_counter()-st)
        if (self.t+self.global_n)==1:
            img = img2[0].transpose(2,0,1) # c w h
        else:
            img = img2.transpose(0,3,1,2) # t c w h
#         print('Processed:',time.perf_counter()-st)
#         st = time.perf_counter()    
        img = torch.from_numpy(img)
#         print('Img2tensor:',time.perf_counter()-st)
#         st = time.perf_counter()    
        label = label[::self.rate,::self.rate]
        if self.tag=='part':
            label[label>self.class_num]=0
        label = torch.from_numpy(label)
#         print('Label2tensor:',time.perf_counter()-st)
#         st = time.perf_counter()    
        label = F.one_hot(label.to(torch.int64),num_classes=self.class_num+1).permute(2,0,1)
#         print('final data:',time.perf_counter()-st)
        return {'path': [ins,frame],'image': img,'label': label}

    def __len__(self):
        return self.num_samples



if __name__ == '__main__':
    from tqdm import tqdm
    import pickle

    dataset = endovis2017('train',fold=0,t=5,rate=1)
    for d in dataset:
        b1 = d
        print(d['image'].shape)
        print(d['label'].shape)
        break
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8,
                                             shuffle=True, num_workers=2,
                                             pin_memory=True, drop_last=True)