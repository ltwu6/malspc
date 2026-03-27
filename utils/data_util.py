import numpy as np
import os
import shutil

def check_degree(alpha):
    res = alpha if alpha<360 else alpha-360
    return res

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]

def check_dir(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def remove_store(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path, ignore_errors=True)
    # check_dir(to_path)
    shutil.copytree(from_path, to_path)

def depthToint16(dMap, minVal=0, maxVal=10):
    #Maximum and minimum distance of interception 
    dMap[dMap>maxVal] = maxVal
    # print(np.max(dMap),np.min(dMap))
    dMap = ((dMap-minVal)*(pow(2,16)-1)/(maxVal-minVal)).astype(np.uint16)
    dMap[dMap<0]=0
    return dMap