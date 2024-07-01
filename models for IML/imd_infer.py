import os
import cv2
import mmcv
import time
import argparse
import pickle
# from sklearn.metrics import roc_auc_score
from mmseg.apis import inference_segmentor, init_segmentor, inference_classifier
from mmcv.runner import wrap_fp16_model
import numpy as np
import time
from tqdm import tqdm
from torch.nn import functional as F
parser = argparse.ArgumentParser(description='Train a segmentor')
# parser.add_argument('--ipt', type=str)
parser.add_argument('--cfg', type=str)
parser.add_argument('--pth', type=str)
parser.add_argument('--abl', type=str)
parser.add_argument('--out', type=str)
# parser.add_argument('--local-rank', type=int, default=0)
parser.add_argument('--sz', type=int, default=768)
args = parser.parse_args()
size_dict = {
    768: (1536, 768),
    1024: (1536, 1024),
    1280: (2048, 1280),
    1536: (3072, 1536),
    1792: (3072, 1792), 
    2048: (3072, 2048),
    2560: (3072, 2560),
    3072: (4096, 3072),
}
this_size = size_dict[args.sz]
map_dict = {}

device = 'cuda:0'

configs = [
    args.cfg,
]
ckpts = [
    args.pth,
]

data_dict = {
    'imdf':('test_datas/IMD20/imgs/', 'test_datas/IMD20/masks/', '.png'),
}
f = open('qes_imd_infer.txt','a+')
# data = args.ipt # 'test_images'#'/home/qcf-568/disk/mmseg/mmseg/mmseg_tianchi/datas/tianchi/test/images'#
# out = './outputs/'+args.out+'_%d'%args.sz
if not (args.out is None):
  if not os.path.exists(args.out):                  
    os.makedirs(args.out)        
  save=True
else:
  save=False
# start_time = time.time()
models = init_segmentor(configs[0], ckpts[0], abl=args.abl, device=device)
for dnm,(data,gtdir,hz) in data_dict.items():
    filenames = os.listdir(data)
    acc=0
    pfk=0
    pau=0
    if save:
        current_time = time.localtime()
        formatted_time = str(time.strftime("%Y-%m-%d %H:%M:%S", current_time))
        save_nm = os.path.join(args.out, dnm+formatted_time)
        if not os.path.exists(save_nm):
            os.makedirs(save_nm)
    ious = []
    ps = []
    rs = []
    fs = []
    ins = 0
    uns = 0
    pds = 0
    gts = 0
    for fi,filename in enumerate(tqdm(filenames)):
      if True:
        single_img = os.path.join(data, filename)
        try:
            single_gt = (cv2.imread(os.path.join(gtdir, filename.split('.')[0] + hz),0)>0).astype(np.uint8)
        except:
            print(fi,filename)
        pred = inference_segmentor(models, single_img)
        # AUC = roc_auc_score(single_gt.reshape(-1,), preds.reshape(-1,))
        # print(pred.shape,single_gt.shape,pred.max(),single_gt.max())
        h,w = single_gt.shape
        h1,w1 = pred.shape
        if ((h1!=h) or (w1!=w)):
            print('rsz', filename)
            single_gt = cv2.resize(single_gt, (w1, h1))
        i2 = (pred * single_gt).sum()
        psum = pred.sum()
        gsum = single_gt.sum()
        u2 = (pred.sum()+single_gt.sum()-i2)
        ious.append((i2/(u2+1e-6)))
        p = (i2/(psum+1e-6))
        r = (i2/(gsum+1e-6))
        ins = (ins + i2)
        uns = (uns + u2)
        pds = (pds + psum)
        gts = (gts + gsum)
        ps.append(p)
        rs.append(r)
        fs.append((2*p*r/(p+r+1e-8)))
        # exit(0)
        pred = (pred*255).astype(np.uint8)
        if save:
            cv2.imwrite(os.path.join(save_nm, filename.split('.')[0] + '.png'), pred) 
      else:
        print('error',filename)
    ious = np.array(ious).mean()
    ps = np.array(ps).mean()
    rs = np.array(rs).mean()
    fs = np.array(fs).mean()
    pu = (ins/(pds+1e-8))
    ru = (ins/(gts+1e-8))
    print(ious, ps, rs, fs) # IoU, Precision, Recall, Fscore
    f.write(args.cfg+'\t'+args.pth+'\t'+str(ious)+'\t'+str(ps)+'\t'+str(rs)+'\t'+str(fs)+'\n')
f.close()
