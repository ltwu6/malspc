import h5py
import numpy as np
import os

src  = '/disk2/wulintai/data/CRN/our_data/test_data.h5'
dst = '/disk2/wulintai/data/CRN/test'
ctgs = ['plane', 'cabinet', 'car', 'chair', 'lamp', 'couch', 'table', 'watercraft']
f = h5py.File(src,'r')
# print(f.keys())
# print(type(f['complete_pcds'][()]))
# print(f['complete_pcds'][()].shape)
# print(f['incomplete_pcds'][()].shape)
print(np.unique(f['labels'][()]))
for i in range(f['complete_pcds'][()].shape[0]):
    gt = f['complete_pcds'][()][i]
    part = f['incomplete_pcds'][()][i]
    id = f['labels'][()][i]
    tax = ctgs[id]
    os.makedirs(os.path.join(dst,'gt',tax),exist_ok=True)
    os.makedirs(os.path.join(dst,'part',tax),exist_ok=True)
    np.save(os.path.join(dst,'gt',tax,str(i)+'.npy'),gt)
    np.save(os.path.join(dst,'part',tax,str(i)+'.npy'),part)

f.close()


















