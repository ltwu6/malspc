import torch
import numpy as np
import os
import torch.nn as nn
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from models.utils import fps_subsample
chamfer_dist = chamfer_3DDist()
from .cdloss import chamfer_distance


def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    # print(f'd1 min:{torch.min(d1)}, d2 min:{torch.min(d2)}')
    # print(f'd1 max:{torch.max(d1)}, d2 max:{torch.max(d2)}')
    #print('shape of d1 and d2: ', d1.shape, d2.shape)
    d1 = torch.mean(torch.sqrt(d1+1e-8))
    d2 = torch.mean(torch.sqrt(d2+1e-8))
    # print(f'd1: {d1}, d2:{d2}')
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1+1e-8))
    return d1


def get_loss(pcds_pred, partial, gt=None, sqrt=False):
    """loss function
    Args
        pcds_pred: List of predicted point clouds. fine and coarse prediction
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1 = pcds_pred # Pc: fine; P1: coarse

    # gt_2 = fps_subsample(gt, P2.shape[1])
    # gt_1 = fps_subsample(gt_2, P1.shape[1])
    # gt_c = fps_subsample(gt_1, Pc.shape[1])

    if gt is not None:
        cdc = CD(Pc, gt)
        partial_matching = 0
    else:
        cdc = 0
        partial_matching = PM(partial, P1)

    # loss_all = (cdc + cd1 + cd2 + cd3 + partial_matching) * 1e3
    losses = [cdc, partial_matching]
    return losses

# negative IoU Loss for silhouette image
def iou(predict, target, eps=1e-6):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + eps
    return (intersect / union).sum() / intersect.nelement()
def iou_loss(predict, target):
    return 1 - iou(predict, target)

def multiview_iou_loss(predicts, targets_a, resize):
    loss = (iou_loss(predicts[:,0,0,:,:], resize(targets_a[:,0,3,:,:])) +
            iou_loss(predicts[:,1,0,:,:], resize(targets_a[:,1,3,:,:])) +
            iou_loss(predicts[:,2,0,:,:], resize(targets_a[:,2,3,:,:])) +
            iou_loss(predicts[:,3,0,:,:], resize(targets_a[:,3,3,:,:])) +
            iou_loss(predicts[:,4,0,:,:], resize(targets_a[:,4,3,:,:])) +
            iou_loss(predicts[:,5,0,:,:], resize(targets_a[:,5,3,:,:])) +
            iou_loss(predicts[:,6,0,:,:], resize(targets_a[:,6,3,:,:])) +
            iou_loss(predicts[:,7,0,:,:], resize(targets_a[:,7,3,:,:]))
            ) / 8.0
    return loss


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        # csize = x.size()[4]
        h_x = x.size()[1]
        w_x = x.size()[2]
        # count_h =  (x.size()[2]-1) * x.size()[3]
        # count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,1:,:]-x[:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,1:]-x[:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*(h_tv+w_tv)/(batch_size*h_x*w_x)

def directed_hausdorff(point_cloud1:torch.Tensor, point_cloud2:torch.Tensor, reduce_mean=True):
    """
    # UHD from MPC: https://github.com/ChrisWu1997/Multimodal-Shape-Completion/blob/master/evaluation/completeness.py
    :param point_cloud1: (B, 3, N)
    :param point_cloud2: (B, 3, M)
    :return: directed hausdorff distance, A -> B
    """

    # print('shape of point cloud 1,2: ', point_cloud1.shape, point_cloud2.shape)
    n_pts1 = point_cloud1.shape[2]
    n_pts2 = point_cloud2.shape[2]

    pc1 = point_cloud1.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2)) # (B, 3, N, M)
    pc2 = point_cloud2.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1)) # (B, 3, N, M)

    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

    shortest_dist, _ = torch.min(l2_dist, dim=2)

    hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )

    if reduce_mean:
        hausdorff_dist = torch.mean(hausdorff_dist)

    return hausdorff_dist


def mmd_cd(pc, gt_dir, pc_name, dst):
    obj_cd = []
    gt_list = os.listdir(gt_dir)
    # gt_list = gt_list[:1]
    for gname in gt_list:
        gt_name = os.path.join(gt_dir, gname, '0.npy')
        gt_pc = np.load(gt_name)
        gt_torch = torch.from_numpy(gt_pc).float().cuda()
        gt_torch = gt_torch.unsqueeze(0)
        cd, _ = chamfer_distance(pc, gt_torch, point_reduction='mean')
        cd *= 1e4
        obj_cd.append(float(cd.detach().cpu()))
    min_objcd = np.min(obj_cd)
    with open(os.path.join(dst), 'a') as mmdf:
        mmdf.write(pc_name+':       '+str(round(min_objcd, 4))+'\n')
    return min_objcd




