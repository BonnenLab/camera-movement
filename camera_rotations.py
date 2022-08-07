from __future__ import print_function
import os
import sys
import scipy.linalg
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from utils.io import mkdir_p
from utils import dydepth as ddlib
from math import acos
import pandas as pd

cudnn.benchmark = False
cur_path = os.path.dirname(__file__)
LOAD_MODEL = './weights/rigidmask-sf/weights.pth' # model path

DATA_PATH = sys.argv[1] # path to video frames
OUTDIR = sys.argv[2] # path of dataframe output

def rotation_mag(rot):
    ''' takes in rotation matrix and returns angle of rotation. '''
    trace = rot[0,0] + rot[1,1] + rot[2,2]
    return acos((trace - 1) / 2)

def rotation_axis(rot):
    ''' takes in rotation matrix and returns axis of rotation. '''
    vals, vecs = scipy.linalg.eig(rot)
    for v in range(len(vals)):
        if vals[v] == 1:
            return vecs[v]
    
# load data
from dataloaders import seqlist as DA
test_left_img, test_right_img, _ = DA.dataloader(DATA_PATH)  
maxh, maxw = [640, 480]

# get input dimensions for model
max_h = int(maxh // 64 * 64)
max_w = int(maxw // 64 * 64)
if max_h < maxh: max_h += 64
if max_w < maxw: max_w += 64
maxh = max_h
maxw = max_w

mean_L = [[0.33,0.33,0.33]]
mean_R = [[0.33,0.33,0.33]]

# construct model, VCN-expansion
from models.VCNplus import VCN
model = VCN([1, maxw, maxh], md=[4,4,4,4,4], fac=1, exp_unc=True)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()


pretrained_dict = torch.load(LOAD_MODEL,map_location='cpu')
mean_L=pretrained_dict['mean_L']
mean_R=pretrained_dict['mean_R']
pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}
model.load_state_dict(pretrained_dict['state_dict'],strict=False)

mkdir_p(OUTDIR)
def main():
    model.eval()
    xs = []
    ys = []
    zs = []
    axes = []
    mags = []
    for inx in range(len(test_left_img)):
        idxname = test_left_img[inx].split('/')[-1].split('.')[0]
        # try to load image pair, if unable to load either, continue
        try:
            if 'jpg' != test_left_img[inx][-3:] or 'jpg' != test_right_img[inx][-3:]:
                continue
            imgL_o = cv2.imread(test_left_img[inx])[:,:,::-1]
            imgR_o = cv2.imread(test_right_img[inx])[:,:,::-1]
        except:
            continue
        
        # resize and prepare frames to be ran through model
        maxh = imgL_o.shape[0]
        maxw = imgL_o.shape[1]
        max_h = int(maxh // 64 * 64)
        max_w = int(maxw // 64 * 64)
        if max_h < maxh: max_h += 64
        if max_w < maxw: max_w += 64
        input_size = imgL_o.shape
        imgL = cv2.resize(imgL_o,(max_w, max_h))
        imgR = cv2.resize(imgR_o,(max_w, max_h))
        imgL_noaug = torch.Tensor(imgL/255.)[np.newaxis].float().cuda()

        # flip channel, subtract mean (common trick in correspondence matching, used to ensure correlation between image patches is only high when patches are similar)
        imgL = imgL[:,:,::-1].copy() / 255. - np.asarray(mean_L).mean(0)[np.newaxis,np.newaxis,:]
        imgR = imgR[:,:,::-1].copy() / 255. - np.asarray(mean_R).mean(0)[np.newaxis,np.newaxis,:]
        imgL = np.transpose(imgL, [2,0,1])[np.newaxis]
        imgR = np.transpose(imgR, [2,0,1])[np.newaxis]

        # modify module according to inputs
        from models.VCNplus import WarpModule, flow_reg
        for i in range(len(model.module.reg_modules)):
            model.module.reg_modules[i] = flow_reg([1,max_w//(2**(6-i)), max_h//(2**(6-i))], 
                            ent=getattr(model.module, 'flow_reg%d'%2**(6-i)).ent,\
                            maxdisp=getattr(model.module, 'flow_reg%d'%2**(6-i)).md,\
                            fac=getattr(model.module, 'flow_reg%d'%2**(6-i)).fac).cuda()
        for i in range(len(model.module.warp_modules)):
            model.module.warp_modules[i] = WarpModule([1,max_w//(2**(6-i)), max_h//(2**(6-i))]).cuda()
        
        # camera intrinsics
        fl = min(input_size[0], input_size[1]) *2
        fl_next = fl
        cx = input_size[1]/2.
        cy = input_size[0]/2.
        bl = 1
        K0 = np.eye(3)
        K0[0,0] = fl
        K0[1,1] = fl
        K0[0,2] = cx
        K0[1,2] = cy
        K1 = K0
        intr_list = [torch.Tensor(inxx).cuda() for inxx in [[fl],[cx],[cy],[bl],[1],[0],[0],[1],[0],[0]]]
        
        intr_list.append(torch.Tensor([input_size[1] / max_w]).cuda()) # delta fx
        intr_list.append(torch.Tensor([input_size[0] / max_h]).cuda()) # delta fy
        intr_list.append(torch.Tensor([fl_next]).cuda())
        disc_aux = [None,None,None,intr_list,imgL_noaug,None]
        
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
        with torch.no_grad():
            # prepare frames to be passed into VCN
            imgLR = torch.cat([imgL,imgR],0)
            model.eval()
            torch.cuda.synchronize()
            
            # VCN (optical flow) -> U-Net (optical expansion) -> NG-RANSAC (essential matrix for image rectifying)
            rts = model(imgLR, disc_aux, None)
            torch.cuda.synchronize()
            '''
            flow -> estimated optical flow
            occ -> uncertainty of optical flow
            fgmask -> prelogits of 0-1 probability foreground vs background
            polarmask -> segmentation label of foreground objects 
            disp -> depth contrast cost (used to address colinear motion degeneracy)
            '''
            flow, occ, _, _, fgmask, _, polarmask, disp = rts
            polarmask = polarmask['mask']   
            polarmask = polarmask[polarmask.shape[0]//2:]
        
        # upsampling
        occ = cv2.resize(occ.data.cpu().numpy(),  (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        fgmask = cv2.resize(fgmask.cpu().numpy(), (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        polarmask= cv2.resize(polarmask, (input_size[1],input_size[0]),interpolation=cv2.INTER_NEAREST).astype(int)
        polarmask[np.logical_and(fgmask>0,polarmask==0)]=-1
        disp= cv2.resize(disp.cpu().numpy(), (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        flow = torch.squeeze(flow).data.cpu().numpy()
        flow = np.concatenate( [cv2.resize(flow[0],(input_size[1],input_size[0]))[:,:,np.newaxis],
                                cv2.resize(flow[1],(input_size[1],input_size[0]))[:,:,np.newaxis]],-1)
        
        # scale flow
        flow[:,:,0] *= imgL_o.shape[1] / max_w
        flow[:,:,1] *= imgL_o.shape[0] / max_h
        
        
        flow = np.concatenate( (flow, np.ones([flow.shape[0],flow.shape[1],1])),-1)
        
        ## depth and scene flow estimation
        mask_input = polarmask
        bgmask = (mask_input == 0) 
        shape = flow.shape[:2]
        x0,y0=np.meshgrid(range(shape[1]),range(shape[0]))
        x0=x0.astype(np.float32)
        y0=y0.astype(np.float32)
        x1=x0+flow[:,:,0]
        y1=y0+flow[:,:,1]
        hp0 = np.concatenate((x0[np.newaxis],y0[np.newaxis],np.ones(x1.shape)[np.newaxis]),0).reshape((3,-1))
        hp1 = np.concatenate((x1[np.newaxis],y1[np.newaxis],np.ones(x1.shape)[np.newaxis]),0).reshape((3,-1))
        
        # valid flow pixels are estimated by the VCN
        valid_mask = np.logical_and(bgmask, occ<0).flatten()
        # use bg + valid pixels to compute R/t
        R01,T01,H01,comp_hp1,E = ddlib.pose_estimate(K0,K1,hp0,hp1,valid_mask,[0,0,0])   
        
        # parallax flow magnitude = rectified optical flow after rotation removal
        parallax = np.transpose((comp_hp1[:2]-hp0[:2]),[1,0]).reshape(x1.shape+(2,))        
        parallax_mag = np.linalg.norm(parallax[:,:,:2],2,2)
        
        # if the average parallax flow magnitude < 2, camera is static
        if parallax_mag[bgmask].mean()<2:
            # static camera
            T01_c = [0,0,0]
        else:
            # determine scale of translation / reconstruction
            aligned_mask,T01_c,ranked_p = ddlib.evaluate_tri(T01,R01,K0,K1,hp0,hp1,disp,occ,bl,inlier_th=0.01,select_th=1.2,valid_mask=valid_mask)
        
        # calculate magnitude of rotation
        mag = rotation_mag(R01)
        # calculate rotation axis
        axis = rotation_axis(R01)
        
        if T01_c is None:
            T01_c = [None, None, None]
        
        # append to lists for saving
        mags.append(mag)
        axes.append(axis)
        xs.append(T01_c[0])
        ys.append(T01_c[1])
        zs.append(T01_c[2])
        
        torch.cuda.empty_cache()
    return xs, ys, zs, mags, axes
                
            

if __name__ == '__main__':
    xs, ys, zs, mags, axes = main()
    df = pd.DataFrame({"X": xs, "Y": ys, "Z": zs, "Magnitude": mags, "Axis": axes})
    df.to_csv("rotations/rotations_" + sys.argv[1][:-1] + ".csv")
