from logging import root
import os
from torchvision import transforms
import os.path
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pickle
import random
import math 
from tqdm import tqdm
import scipy.io as sio
import sys
from io_util import read_pcd, read_mat
from data_util import check_degree, resample_pcd
import cv2


class TrainDataLoader(Dataset):
    def __init__(self,filepath,data_path='', pc_input_num=2048,category='all'):
        super(TrainDataLoader,self).__init__()
        """
        Args:
        filepath: list of dataset
        data_path: root path of training and test data
        status: 'train', 'valid' or 'test'

        """
        self.cat_map = {
            'plane':'02691156',
            'cabinet':'02933112',
            'car':'02958343',
            'chair':'03001627',
            'lamp':'03636649',
            'couch':'04256520',
            'table':'04379243',
            'watercraft':'04530566'
        }
        self.pc_input_num = pc_input_num
        self.filepath = os.path.join(filepath, 'train.list')
        self.data_path = data_path
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category
        self.MAX_DIST = 1.2
        
        with open(self.filepath,'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line.strip())
                line = f.readline()
        
        self.incomplete_path = os.path.join(self.data_path,'part') 
        self.rendering_path = os.path.join(self.data_path,'depth_raw')
        self.param_path = os.path.join(self.data_path, 'params_npy')
        self.silh_path = os.path.join(self.data_path, 'mask')
        for key in self.filelist:
            if category !='all':
                if key.split('/')[0]!= self.cat_map[category]:
                    continue
            self.cat.append(key.split('/')[0])
            self.key.append(key)

        self.d_len = len(self.key)
        self.d_id = self.d_len
        # print('d len: ', self.d_len) # 1322
        self.dep_index = list(range(self.d_len))
        np.random.shuffle(self.dep_index)
        
    
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ]) # shape: [C, H, W]; value: 0-1

        print(f'train data num: {len(self.key)}')

    def get_dep_index(self):
        if self.d_id >= self.d_len:
            self.d_id = 0
            np.random.shuffle(self.dep_index)
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print('sn index', self.sn_idx)
        res = self.dep_index[self.d_id]
        # print('shapenet index', res)
        self.d_id += 1
        # print('d id: ', self.d_id)
        # print('dep index: ', res)
        return res 

    def __getitem__(self, idx):
        key = self.key[idx].strip()
        # print('key: ', key)
        rand_id = self.get_dep_index()
        key_dep = self.key[rand_id]
        pc_part_path = os.path.join(self.incomplete_path,key,'0.npy')
        view_path = os.path.join(self.rendering_path,key_dep,'0.npy') # e.g:categ/obj/
        silh_path = os.path.join(self.silh_path, key,'0.npy')
        param_path = os.path.join(self.param_path, key,'0.npy')
        depth_view = np.load(view_path)
        silh_mask = np.load(silh_path)
        param = np.load(param_path)
        pc_part = np.load(pc_part_path)
        if pc_part.shape[0]!=self.pc_input_num:
            pc_part = resample_pcd(pc_part, self.pc_input_num)


        return torch.from_numpy(depth_view).float(), torch.from_numpy(pc_part).float(),\
               torch.from_numpy(silh_mask).float(), torch.from_numpy(param).float(), key

    def __len__(self):
        return len(self.key)

class TestDataLoader(Dataset):
    def __init__(self,filepath,data_path,status,pc_input_num=2048,category='all'):
        super(TestDataLoader,self).__init__()
        """
        Args:
        filepath: list of dataset
        data_path: root path of training and test data
        status: 'valid' or 'test'

        """
        self.cat_map = {
            'plane':'02691156',
            'cabinet':'02933112',
            'car':'02958343',
            'chair':'03001627',
            'lamp':'03636649',
            'couch':'04256520',
            'table':'04379243',
            'watercraft':'04530566'
        }
        self.pc_input_num = pc_input_num
        self.status = status
        self.data_path = data_path
        self.filepath = os.path.join(filepath, self.status+'.list')
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category
        self.MAX_DIST = 1.75
        
        with open(self.filepath,'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line.strip())
                line = f.readline()
        
        self.incomplete_path = os.path.join(self.data_path,'part') 
        self.gt_path = os.path.join(self.data_path,'gt_16384')
        self.rendering_path = os.path.join(self.data_path,'depth_raw')      
        self.param_path = os.path.join(self.data_path, 'params_npy')
        for key in self.filelist:
            if category !='all':
                if key.split('/')[0]!= self.cat_map[category]:
                    continue
            self.cat.append(key.split('/')[0])
            self.key.append(key)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        print(f'{self.status} data num: {len(self.key)}')


    def __getitem__(self, idx):
        key = self.key[idx].strip()
        pc_part_path = os.path.join(self.incomplete_path,key,'0.npy')
        pc_path = os.path.join(self.gt_path,key,'0.npy')
        view_path = os.path.join(self.rendering_path,key,'0.npy') # e.g:categ/obj/
        param_path = os.path.join(self.param_path, key,'0.npy')
        depth_view = np.load(view_path)
        param = np.load(param_path)
        pc = np.load(pc_path)
        pc_part = np.load(pc_part_path)
        if pc_part.shape[0]!=self.pc_input_num:
            pc_part = resample_pcd(pc_part, self.pc_input_num)

        return torch.from_numpy(depth_view).float(), torch.from_numpy(param).float(),\
            torch.from_numpy(pc_part).float(), torch.from_numpy(pc).float(), key



    def __len__(self):
        return len(self.key)

