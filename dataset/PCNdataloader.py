import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from utils.logger import *
from torch.utils.data import Dataset
from .io import check_degree, resample_pcd
import torch

# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

# class PCN(data.Dataset):
#     # def __init__(self, data_root, subset, class_choice = None):
#     def __init__(self, PARTIAL_POINTS_PATH,COMPLETE_POINTS_PATH,CATEGORY_FILE_PATH,N_POINTS,subset,categories):
#         self.partial_points_path = PARTIAL_POINTS_PATH
#         self.complete_points_path = COMPLETE_POINTS_PATH
#         self.category_file = CATEGORY_FILE_PATH
#         self.npoints = N_POINTS
#         self.subset = subset
#         self.categories = categories

#         # Load the dataset indexing file
#         self.dataset_categories = []
#         with open(self.category_file) as f:
#             self.dataset_categories = json.loads(f.read())
#             self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_name'] in self.categories]

#         self.n_renderings = 8 if self.subset == 'train' else 1
#         self.file_list = self._get_file_list(self.subset, self.n_renderings)
#         self.transforms = self._get_transforms(self.subset)

#     def _get_transforms(self, subset):
#         # if subset == 'train':
#         #     return data_transforms.Compose([{
#         #         'callback': 'RandomSamplePoints',
#         #         'parameters': {
#         #             'n_points': 2048
#         #         },
#         #         'objects': ['partial', 'gt']
#         #     }, {
#         #         'callback': 'RandomMirrorPoints',
#         #         'objects': ['partial', 'gt']
#         #     },{
#         #         'callback': 'ToTensor',
#         #         'objects': ['partial', 'gt']
#         #     }])
#         # else:
#             return data_transforms.Compose([{
#                 'callback': 'RandomSamplePoints',
#                 'parameters': {
#                     'n_points': 2048
#                 },
#                 'objects': ['partial']
#             }, {
#                 'callback': 'ToTensor',
#                 'objects': ['partial', 'gt']
#             }])

#     def _get_file_list(self, subset, n_renderings=1):
#         """Prepare file list for the dataset"""
#         file_list = []

#         for dc in self.dataset_categories:
#             print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
#             samples = dc[subset]

#             for s in samples:
#                 file_list.append({
#                     'taxonomy_id':
#                     dc['taxonomy_id'],
#                     'model_id':
#                     s,
#                     'partial_path': [
#                         self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
#                         for i in range(n_renderings)
#                     ],
#                     'gt_path':
#                     self.complete_points_path % (subset, dc['taxonomy_id'], s),
#                 })

#         print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
#         return file_list

#     def __getitem__(self, idx):
#         sample = self.file_list[idx]
#         data = {}
#         rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0
        
#         for ri in ['partial', 'gt']:
#             file_path = sample['%s_path' % ri]
#             if type(file_path) == list:
#                 file_path = file_path[rand_idx]
#             data[ri] = IO.get(file_path).astype(np.float32)

#         assert data['gt'].shape[0] == self.npoints

#         if self.transforms is not None:
#             # print('using transform')
#             data = self.transforms(data)

#         return sample['taxonomy_id'], sample['model_id'], sample['partial_path'][rand_idx], (data['partial'], data['gt'])

#     def __len__(self):
#         return len(self.file_list)
    
# class PCN_Test(data.Dataset):
#     # def __init__(self, data_root, subset, class_choice = None):
#     def __init__(self, PARTIAL_POINTS_PATH,COMPLETE_POINTS_PATH,CATEGORY_FILE_PATH,N_POINTS,subset,categories):
#         self.partial_points_path = PARTIAL_POINTS_PATH
#         self.complete_points_path = COMPLETE_POINTS_PATH
#         self.category_file = CATEGORY_FILE_PATH
#         self.npoints = N_POINTS
#         self.subset = subset
#         self.categories = categories

#         # Load the dataset indexing file
#         self.dataset_categories = []
#         with open(self.category_file) as f:
#             self.dataset_categories = json.loads(f.read())
#             self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_name'] in self.categories]

#         self.n_renderings = 8
#         self.file_list = self._get_file_list(self.subset, self.n_renderings)
#         self.transforms = self._get_transforms(self.subset)

#     def _get_transforms(self, subset):
#         # if subset == 'train':
#         #     return data_transforms.Compose([{
#         #         'callback': 'RandomSamplePoints',
#         #         'parameters': {
#         #             'n_points': 2048
#         #         },
#         #         'objects': ['partial', 'gt']
#         #     }, {
#         #         'callback': 'RandomMirrorPoints',
#         #         'objects': ['partial', 'gt']
#         #     },{
#         #         'callback': 'ToTensor',
#         #         'objects': ['partial', 'gt']
#         #     }])
#         # else:
#             return data_transforms.Compose([{
#                 'callback': 'RandomSamplePoints',
#                 'parameters': {
#                     'n_points': 2048
#                 },
#                 'objects': ['partial']
#             }, {
#                 'callback': 'ToTensor',
#                 'objects': ['partial', 'gt']
#             }])

#     def _get_file_list(self, subset, n_renderings=1):
#         """Prepare file list for the dataset"""
#         file_list = []

#         for dc in self.dataset_categories:
#             print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
#             samples = dc[subset]

#             for s in samples:
#                 for i in range(n_renderings):
#                     file_list.append({
#                         'taxonomy_id':
#                         dc['taxonomy_id'],
#                         'model_id':
#                         s,
#                         'partial_path': [
#                             self.partial_points_path % (subset, dc['taxonomy_id'], s, i)],
#                         'gt_path':
#                         self.complete_points_path % (subset, dc['taxonomy_id'], s),
#                     })

#         print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
#         return file_list

#     def __getitem__(self, idx):
#         sample = self.file_list[idx]
#         data = {}
#         # rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0
        
#         for ri in ['partial', 'gt']:
#             file_path = sample['%s_path' % ri]
#             if type(file_path) == list:
#                 file_path = file_path[0]
#             data[ri] = IO.get(file_path).astype(np.float32)

#         assert data['gt'].shape[0] == self.npoints

#         if self.transforms is not None:
#             # print('using transform')
#             data = self.transforms(data)

#         return sample['taxonomy_id'], sample['model_id'], sample['partial_path'][0], (data['partial'], data['gt'])

#     def __len__(self):
#         return len(self.file_list)
    

class TrainDataLoader(Dataset):
    def __init__(self,data_path,category='all', gt_pnum = 2048, resolution = 128):
        super(TrainDataLoader,self).__init__()
        """
        Args:
        filepath: list of dataset:./txt_files
        data_path: root path of training data:/disk1/lintaiwu/data/PCN/ShapeNetCompletion_npy/train
        status: 'train', 'valid' or 'test'

        """
        # self.filepath = os.path.join(filepath, 'shapenet_train.txt')
        self.gt_num = gt_pnum
        self.resolution = resolution
        self.filelist = []
        self.cat = []
        self.category = category
        self.cat_map = {
            'bag':'02773838',
            'lamp':'03636649',
            'bathtub':'02808440',
            'bed':'02818832',
            'basket':'02801938',
            'printer':'04004475',
            'laptop':'03642806',
            'bench':'02828884'
        }
        # with open(self.filepath,'r') as f:
        #     line = f.readline()
        #     while (line):
        #         for i in range(4,8):
        #             self.filelist.append(line.strip()+os.sep+'input_'+str(i)+'.npy')
        #         line = f.readline()
        self.category = list(self.cat_map.values()) if category == 'all' else [category]
        for ctg in self.category:
            objs = os.listdir(os.path.join(data_path,'partial',ctg))
            for obj in objs:
                for i in range(8):
                    self.filelist.append(ctg+'/'+obj+'/'+str(i).rjust(2,'0')+'.npy')
        print('length of filelist: ', len(self.filelist))
        self.incomplete_path = os.path.join(data_path,'partial') # 'train/new_part' if use new partial
        self.gt_path = os.path.join(data_path,'complete')
        print(f'train data num: {len(self.filelist)}')

    # def normalize_point(self, pc):
    #         pc = pc-0.5#
    #         return pc

    def __getitem__(self, idx):
        key = self.filelist[idx]
        #print('key: ', key)
        pc_part_path = os.path.join(self.incomplete_path,key)
        pc_path = os.path.join(self.gt_path,key.split('/')[0],key.split('/')[1]+'.npy')
        pc_part = np.load(pc_part_path)
        pc_gt = np.load(pc_path)
        if pc_gt.shape[0]!=self.gt_num:
            pc_gt = resample_pcd(pc_gt, self.gt_num)
        if pc_part.shape[0]!=self.gt_num:
            pc_part = resample_pcd(pc_part, self.gt_num)
        # pc_part = self.normalize_point(pc_part)
        # pc_gt = self.normalize_point(pc_gt)
        # print('shape of gt and part: ', pc_gt.shape, pc_part.shape)#(4096, 3) (2048, 3)
        return torch.from_numpy(pc_part).float(), torch.from_numpy(pc_gt).float(), key

    def __len__(self):
        return len(self.filelist)




































