#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Lintai Wu

import ast
from operator import inv
from pyexpat import features
from matplotlib import gridspec
from pandas import concat
import numpy as np
import itertools
from typing import cast
import random
import math
from .P2C import PCN_Van #PCN_Van
import torch
import torch.nn as nn
from torch.autograd import Variable
from .discriminator import Discriminator
from models.utils import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, Transformer,\
     Conv1d, Conv2d, UpDownUp_Folding, MLP
from utils.cdloss import chamfer_distance, chamfer_distance_sqrt, chamfer_distance_sqrtp
from utils.loss_utils import TVLoss
import torch.nn.functional as F
# from torchmetrics.image import TotalVariation
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from render_tools.rasterizer import PointsRasterizer_AdaRad
from render_tools.renderer import PointsRenderer_AdaRad
# DEVICE = torch.device('cuda:0')
DEVICE = 'cuda:0'


class Renderer(nn.Module):
    def __init__(self, img_size=224, FMM = 35.0, SENSOR_SIZE_MM = 32.0, MAX_DIST = 1.2, ):
        """Renderer that with density-aware depth radius
        """
        super(Renderer, self).__init__()
        # rendering params
        self.FMM = FMM
        self.img_w = img_size
        # self.ratio = self.img_w/224.0
        self.ratio = 1.0
        self.SENSOR_SIZE_MM = SENSOR_SIZE_MM
        self.fuv = self.FMM * self.img_w / self.SENSOR_SIZE_MM
        self.MAX_DIST = MAX_DIST
        self.device = torch.device("cuda:0")
        self.raster_settings_dep = PointsRasterizationSettings(
                image_size=self.img_w,
                radius = 0.012, 
                points_per_pixel = 16, 
                bin_size=0
            )
        self.raster_settings_sil = PointsRasterizationSettings(
                image_size=self.img_w,
                radius = 0.008, 
                points_per_pixel = 8, 
                bin_size=0
            )

    def forward(self, point_cloud, param, d_flag=True, adasil_flag=False):
        """
        args:
        point_cloud: b,n,6
        param: b,8,3
        """
        bs, point_num = point_cloud.shape[0], point_cloud.shape[1]
        point_cloud = Pointclouds(points=point_cloud, features=torch.ones(size=(bs, point_num, 1)).to(DEVICE)*255)
        R0, T0 = look_at_view_transform(dist=1.2, elev=param[:,1], azim=param[:,0], degrees=True)
        cameras0 = PerspectiveCameras(device=self.device, R=R0, T=T0,
                                     focal_length=(self.fuv), principal_point=((self.img_w/2.0,self.img_w/2.0),),
                                     image_size=((self.img_w, self.img_w),), in_ndc=False)
        rasterizer0 = PointsRasterizer(cameras=cameras0, raster_settings=self.raster_settings_sil)
        renderer0 = PointsRenderer(
            rasterizer=rasterizer0,
            compositor=AlphaCompositor()
        )
        sihlt_map = renderer0(point_cloud)
        if not d_flag:
            return sihlt_map
        else: 
        ### render depth
            elev_d = torch.rand(size=(bs,),device=self.device) *60-30
            # elev = torch.from_numpy(elev).float().cuda()
            azim_d = torch.rand(size=(bs,),device=self.device)*300+param[:,0]+30 # different from the input view point
            # print('shape of param: ', param.shape) # torch.Size([4, 2])
            # print('shape of azim: ', azim.shape) # torch.Size([4])
            # print('shape of elev: ', elev.shape) # torch.Size([4])
            # print('start:')
            R_d, T_d = look_at_view_transform(dist=1.2, elev=elev_d, azim=azim_d, degrees=True)
            # print('................')
            cameras_d = PerspectiveCameras(device=self.device, R=R_d, T=T_d,
                                        focal_length=(self.fuv), principal_point=((self.img_w/2.0,self.img_w/2.0),),
                                        image_size=((self.img_w, self.img_w),), in_ndc=False)
            
            rasterizer_d = PointsRasterizer(cameras=cameras_d, raster_settings=self.raster_settings_dep)
            fragments = rasterizer_d(point_cloud)

            depth_ini = fragments.zbuf[:,:,:,0]
            depth_clone = depth_ini.clone().detach()

            area = torch.sum((depth_clone>0).int(), dim=(2,1)) # b 1
            # print('shape of area: ', area.shape) # [b]
            pa_radius = (area/16384.0)*0.057 
            # plane: 057 cabinet: 030 chair:0486 car:052 lamp:093 couch:049 table:0464
            # watercraft: 085
            pa_radius = pa_radius.unsqueeze(-1).expand(-1,16384).contiguous() # [16, 16384]
            # print('shape of pa_radius: ', pa_radius.shape) # b n
            rasterizer_ada = PointsRasterizer_AdaRad(cameras=cameras_d, raster_settings=self.raster_settings_dep)
            fragments_ada = rasterizer_ada(point_cloud,pa_radius)

            depth_img = fragments_ada.zbuf[:,:,:,:8]

            dist_map = fragments_ada.dists[:,:,:,:]
            # dist_map = torch.pow(dist_map,2)
            dist_top = dist_map[:,:,:,:8]

            return sihlt_map, depth_img, dist_top




class TestLoss(nn.Module):
    def __init__(self):
        super(TestLoss, self).__init__()
        self.loss_name = ['cd_fine']
        self.loss_num = len(self.loss_name)
        self.cd = chamfer_distance
        
    def forward(self,res, gt):
        cd_loss = self.cd(res,gt,point_reduction='mean')[0]
        loss = [cd_loss]
        return loss


### 2048-16384 coarse to fine
class OUR(nn.Module):
    def __init__(self):
        super().__init__()
        """
        
        """
        self.G = PCN_Van()#PCN_S16384()#P2C()
        self.D = Discriminator(1,224)
        self.render=Renderer()
        self.loss = ModelLoss()
        self.loss_test = TestLoss()
        
    # forward function for hier_folding_pe decoder
    def forward(self, pc, input_d, param):
        coarse_output = self.G(pc)
        coarse, fine = coarse_output[0], coarse_output[1]
        feat = coarse_output[2]
        dense, normal = coarse_output[3], coarse_output[4]
        sil_map_c = self.render(coarse, param, d_flag=False, adasil_flag=False)
        sil_map_f, depth_f, dist_f = self.render(fine, param)
        dg,dr = self.D(depth_f[:,:,:,0].unsqueeze(1),input_d.unsqueeze(1))
       
        return  coarse, fine, feat, sil_map_c, sil_map_f,depth_f,dist_f, dense, normal, dg,dr

class ModelLoss(nn.Module):
    def __init__(self,use_cuda=True,bs=16):
        super(ModelLoss, self).__init__()
        self.loss_name = ['loss_g','loss_d']
        self.loss_num = len(self.loss_name)
        self.l1_loss = torch.nn.L1Loss()
        self.adversarial_loss = torch.nn.MSELoss()
        self.l2_loss = torch.nn.MSELoss(reduction='mean')
        self.l2_loss_partial = torch.nn.MSELoss(reduction='none')
        self.partcd = chamfer_distance_sqrtp
        self.latent_criterion = nn.SmoothL1Loss(reduction='mean')
        self.l1cd = chamfer_distance_sqrt
        
        self.Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.valid = Variable(self.Tensor(bs, 1).fill_(1.0), requires_grad=False)
        self.fake = Variable(self.Tensor(bs, 1).fill_(0.0), requires_grad=False)
    def density_loss(self, x):
        x1 = x.unsqueeze(1)
        x2 = x.unsqueeze(2)
        diff = (x1-x2).norm(dim=-1)
        diff, idx = diff.topk(16, largest=False)
        # print(idx.shape)
        loss = diff[:,:,1:].mean(2).std(1)
        return loss
    
    def forward(self, outputs, in_map, in_pc):
        """
        p0: b n 3
        """
        p_c, p_f, smap_c, smap_f, dmap_f, dist_f, dense, normal, dg,dr = \
            outputs[0],outputs[1], outputs[3]/255.0,outputs[4]/255.0, outputs[5], outputs[6], outputs[7], outputs[8], outputs[9], outputs[10]
        # rebuid = outputs[-1]
        # pred_nbrs,part_nbrs = rebuid[0], rebuid[1]
        B = p_c.shape[0]
        # print('shape of inpc: ', in_pc.shape) # [32, 2048, 3]
        ### aul loss
        ## dense loss
        dense_loss = torch.mean(dense.std(1))
        dense_loss_f = torch.mean(self.density_loss(fps_subsample(p_f,2048)))
        # # print('dense loss: ', dense_loss_f) # 0.0044 0.0071 0.0099
        aux_loss = dense_loss+dense_loss_f#+depth_loss
        # shape_recon_loss =  self.l1cd(part_nbrs.reshape(B, -1, 3), pred_nbrs,point_reduction='mean')[0]#
        ### partial cd loss
        shape_recon_loss_c = self.partcd(in_pc, p_c, point_reduction='mean')[0]
        shape_recon_loss_f = self.partcd(in_pc, p_f, point_reduction='mean')[0]
        shape_recon_loss = shape_recon_loss_c + shape_recon_loss_f
        g_cd_loss = shape_recon_loss

        ### silhouette rendering loss
        mask_c = smap_c.clone().detach()
        mask_c = mask_c>0
        g_render_loss_c = torch.mean(self.l2_loss_partial(smap_c,in_map.unsqueeze(-1))*mask_c)
        # g_render_loss_c = self.l2_loss(smap_c,in_map.unsqueeze(-1))
        g_render_loss_f = self.l2_loss(smap_f,in_map.unsqueeze(-1))
        g_render_loss = g_render_loss_c + g_render_loss_f
        # print('g rener loss: ', g_render_loss) # 0.2861
        # print('g_cd_loss: ', g_cd_loss) # 0.4081
        # print('aux_loss: ', aux_loss) # 0.0130
        ## advers loss for generation
        g_advers_loss = self.adversarial_loss(dg,self.valid)
        # print('g advers loss: ', g_advers_loss) # 0.0989 

        g_loss = 1.0*(g_cd_loss+g_render_loss + aux_loss) + 1.0*g_advers_loss

        ### discriminitive loss
        real_loss = self.adversarial_loss(dr,self.valid)
        fake_loss = self.adversarial_loss(dg,self.fake)
        d_loss = 0.5*(real_loss + fake_loss)
        # print('dloss: ', d_loss) # 0.9908
        loss = [g_loss, d_loss]

        return loss





    









