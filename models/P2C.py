import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.cdloss import chamfer_distance, chamfer_distance_sqrt, chamfer_distance_sqrtp
from extensions.pointops.functions import pointops
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from timm.models.layers import trunc_normal_
from utils.logger import *
from models.utils import fps_subsample

class Encoder(nn.Module):
    def __init__(self, feat_dim):
        """
        PCN based encoder
        """
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, feat_dim, 1)
        )

    def forward(self, x):
        bs, n, _ = x.shape
        feature = self.first_conv(x.transpose(2, 1))  # B 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # B 512 n
        feature = self.second_conv(feature)  # B 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B 1024
        return feature_global

class Decoder_C2F(nn.Module):
    def __init__(self, latent_dim=1024, num_output=2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_output = num_output

        ### for normal 
        self.neighborhood_size = 32
        self.support = 24

        self.mlp1 = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 3 * self.num_output)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(1024+3+2+32,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,3,1)
        )
        self.proj_dense_c = nn.Conv1d(15,128,1)
        self.proj_dense_i = nn.Conv1d(15,128,1)
        self.proj_norm_c = nn.Conv1d(24,128,1)
        self.proj_norm_i = nn.Conv1d(24,128,1)

        self.tanh = nn.Tanh()
        self.cos = nn.CosineSimilarity(dim=3, eps=1e-6)
        self.up_sampler = nn.Upsample(scale_factor=8,mode='linear',align_corners=True)
    def density_local(self, x, up_flag=False): 
        x1 = x.unsqueeze(1)
        x2 = x.unsqueeze(2)
        diff_3d = (x1-x2) # b n n 3
        # print('shape of diff3d: ', diff_3d.shape) # [b, 2048, 2048, 3]
        diff = diff_3d.norm(dim=-1) # b n n
        # print('shape of diff: ', diff.shape) # [b, 2048, 2048]
        diff, idx = diff.topk(16, largest=False) # b n 16
        # print('shape of diff idx top k: ', diff.shape, idx.shape)  # [b, 2048, 16]     [b, 2048, 16]   
        den = diff[:,:,1:]
        # print('shape of den: ', den.shape) # [16, 2048, 15]
        if up_flag:
            new_idx = idx.unsqueeze(-1).expand(-1,-1,-1,3) # b n 16 3
            diff3d_top = torch.gather(diff_3d, dim=2, index=new_idx[:,:,:9, :])/2.0 # b n 9 3
            # print('shape of diff3dtop: ', diff3d_top.shape) # [b, 2048, 9, 3]
            upsample_points = x.unsqueeze(2).expand(-1,-1,8,-1)+ diff3d_top[:,:,1:,:] # b n 8 3
            # print('shape of uppoints: ', upsample_points.shape) # [b, 2048, 8, 3]
            upsample_points = torch.flatten(upsample_points,start_dim=1, end_dim=2) # b 16384 3
            # print('shape of uppoints after: ', upsample_points.shape) # [b, 16384, 3]
            return den, den.mean(2), upsample_points
        else:
            return den, den.mean(2)
    
    def normal_local(self, xyz):
        normals = estimate_pointcloud_normals(xyz, neighborhood_size=self.neighborhood_size)
        # print('shape of normals: ', normals.shape) # [16, 2048, 3]
        idx = pointops.knn(xyz, xyz, self.support)[0] # 
        # print('shape of idx: ', idx.shape) # [16, 2048, 24]
        neighborhood = pointops.index_points(normals, idx)
        # print('shape of neigh: ', neighborhood.shape) # [16, 2048, 24, 3]
        cos_similarity = self.cos(neighborhood[:, :, 0, :].unsqueeze(2), neighborhood)
        penalty = 1 - cos_similarity
        return penalty, penalty.std(-1)

    def forward(self, z, input):
        bs = z.size(0)

        coarse_van = self.mlp1(z).reshape(bs, -1, 3).contiguous()  #  B n 3
        coarse = self.tanh(coarse_van)*0.5
        pnum = coarse.shape[1]*8

        ### compute density and normals
        # coarse_fps = fps_subsample(coarse, 2048)
        dense_all, dense = self.density_local(coarse) # [32, 2048, 15] [32, 2048 ]
        normals_all, normals = self.normal_local(coarse) # [32, 2048, 24] [32, 2048 ]
        ## for input
        dense_input_all, dense_input = self.density_local(input) # [32, 2048]
        normals_input_all, normals_input = self.normal_local(input) # [32, 2048]

        dense_all_proj = self.proj_dense_c(dense_all.transpose(2,1)) # b 128 2048
        dense_input_proj = self.proj_dense_i(dense_input_all.transpose(2,1)) # b 128 2048
        normals_all_proj = self.proj_norm_c(normals_all.transpose(2,1)) # b 128 2048
        normals_input_proj = self.proj_norm_i(normals_input_all.transpose(2,1)) # b 128 2048
        # print('shape of dense and normal all and input proj: ', dense_all_proj.shape, dense_input_proj.shape, normals_all_proj.shape, normals_input_proj.shape) # [16, 128, 2048]
        ## compute weight for dense and select top k as feature
        weight_dense = torch.matmul(dense_all_proj.transpose(2,1), dense_input_proj) # b 2048 2048 
        # print('shape of weight_dense: ', weight_dense.shape) # [16, 2048, 2048]       
        weight_dense, idx_dense = weight_dense.softmax(dim=-1).topk(16)  # b 2048 16       
        dense_ref = dense_input.unsqueeze(-1).expand(-1,-1,2048).transpose(2,1) # b 2048 2048
        dense_topk = torch.gather(dense_ref,dim=-1, index=idx_dense).transpose(2,1) # b 16 2048 
        
        ## compute weight for normal and select top k as feature
        weight_normal = torch.matmul(normals_all_proj.transpose(2,1), normals_input_proj) # b 2048 2048
        # print('shape of weight_normal: ', weight_normal.shape) # [16, 2048, 2048]
        weight_normal, idx_normal = weight_normal.softmax(dim=-1).topk(16)  # b 2048 16
        normal_ref = normals_input.unsqueeze(-1).expand(-1,-1,2048).transpose(2,1) # b 2048 2048
        normal_topk = torch.gather(normal_ref,dim=-1, index=idx_normal).transpose(2,1) # b 16 2048 
        # print('shape of weight top dense: ', weight_dense.shape,weight_normal.shape) # [16, 2048, 16]
        # print('shape of dense ref: ', dense_ref.shape,normal_ref.shape) # [16, 2048, 2048]
        # print('shape of dnese top k: ', dense_topk.shape, normal_topk.shape) # [16, 16, 2048]

        _,_, point_feat = self.density_local(coarse_van, up_flag=True)
        point_feat = point_feat.transpose(2,1) # b 3 16384  
        # print('shape of point feat: ', point_feat.shape) # [16, 3, 16384]

        ### for 2048 points
        dense_feat = dense.unsqueeze(1).expand(-1, 8, -1) # b 8 2048
        dense_feat = dense_feat.reshape(bs,pnum).unsqueeze(1) # [b, 1, 16384]
        # print('shape of dense feat: ', dense_feat.shape) # [16, 1, 16384]
        dense_input_feat = dense_topk.unsqueeze(-1).expand(-1, -1, -1, 8) # b 16 2048 8
        # print('shape of dense_input_feat: ', dense_input_feat.shape) # [16, 16, 2048, 8]
        dense_input_feat = torch.flatten(dense_input_feat, start_dim=2, end_dim=-1)
        # print('shape of dense_input_feat after: ', dense_input_feat.shape) # [16, 16, 16384]

        normal_feat = normals.unsqueeze(1).expand(-1,8,-1)
        normal_feat = normal_feat.reshape(bs,pnum).unsqueeze(1) # [b, 1, 16384]
        # print('shape of normal_feat: ', normal_feat.shape) # [16, 1, 16384]
        normal_input_feat = normal_topk.unsqueeze(-1).expand(-1, -1, -1, 8) # b 16 2048 8
        # print('shape of normal_input_feat: ', normal_input_feat.shape) # [16, 16, 2048, 8]
        normal_input_feat = torch.flatten(normal_input_feat, start_dim=2, end_dim=-1)
        # print('shape of normal_input_feat after: ', normal_input_feat.shape) # [16, 16, 16384]

        feature_global = z.unsqueeze(2).expand(-1,-1,pnum) # B 1024 N
        # print('shape of featgobal: ', feature_global.shape)
        feat = torch.cat([dense_input_feat, normal_input_feat,feature_global, dense_feat, normal_feat, point_feat], dim=1) # B 1061 16384
        # feat = torch.cat([feature_global, point_feat], dim=1) # B 1027 N

        offset = self.final_conv(feat) 

        ### for coarse_van
        point_atanh =  point_feat + offset  # B 3 N
        fine = self.tanh(point_atanh)*0.5
        fine = fine.transpose(1,2).contiguous()# b n 3
        return coarse, fine, dense, normals

class PCN_Van(nn.Module):
    def __init__(self):
        super().__init__()
        # define parameters
        self.feat_dim = 1024
        self.n_points = 2048
 
        self.encoder = Encoder(self.feat_dim)
        self.generator = Decoder_C2F(latent_dim=self.feat_dim, num_output=self.n_points)


        self.apply(self._init_weights)
        # init loss
        # self._get_lossfnc_and_weights(config)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, partial):
        # group points
        B, _, _ = partial.shape
        feat = self.encoder(partial)
        # print('shape of feat: ', feat.shape) # [32, 1024]

        coarse, fine, den, norm = self.generator(feat, partial)

        return coarse, fine, feat, den, norm
  
