import sys
sys.path.append("..")

import torch
import os
import cv2
from utils.config import*
import time
import numba

def if_in(box,point):
    #box:[x y x y]
    if point[0]>=box[0] and point[0]<=box[2] and point[1]>=box[1] and point[1]<=box[3]:
        return True
    else:
        return False

#xywh xyxy
import numpy as np
def xywh2xyxy(x):
    # Convert 1x4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y
def xyxy2xywh(x):
    # Convert 1x4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y

#@numba.jit(nopython=True)
def xywh2xyxyn(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def xyxy2xywhn(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def cut_outside(x):
    #cut the outside part of 1x4 xyxy
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = max(0,x[0])
    y[1] = max(0,x[1])
    y[2] = min(1,x[2])
    y[3] = min(1,x[3])
    return y

def box2axis(box):
    #xyxybox
    return [(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])]

def box_iou(box1, box2):

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def box_iou_np(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    #inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def calculate(bound, mask):
    x, y, w, h = bound
    area = mask[y:y+h, x:x+w]
    pos = area > 0 + 0
    score = np.sum(pos)/(w*h)
    return score

def creat_mask(frame_gray, label, fw, fh):
    #create the feature mask of frame[index]
    #label: box of the current frame, nx4
    mask_use = np.zeros_like(frame_gray)

    for b in label:
        #print(int(b[0]))
        #print(b[1])
        bx1 = int((b[0]-0.5*b[2])*fw)
        bx2 = int((b[0]+0.5*b[2])*fw)
        by1 = int((b[1]-0.5*b[3])*fh)
        by2 = int((b[1]+0.5*b[3])*fh)
        mask_use[by1:by2,bx1:bx2] = 255
    return mask_use


def creat_matrix(label, feature, fw, fh):

    if len(feature.shape) == 3:
        feature = feature[:,0,:]
        
    #t1 = time.time()
    matrix = np.full((len(label), len(feature)), 200, dtype=float)
    boxes = xywh2xyxyn(label)
    boxes[:,0] *= fw
    boxes[:,1] *= fh
    boxes[:,2] *= fw
    boxes[:,3] *= fh
    
    
    in_matrix = np.logical_and(
        feature[:, 0][:, None] >= boxes[:, 0][None, :], 
        feature[:, 0][:, None] <= boxes[:, 2][None, :]
    ) & np.logical_and(
        feature[:, 1][:, None] >= boxes[:, 1][None, :], 
        feature[:, 1][:, None] <= boxes[:, 3][None, :]
    )
    
    matrix[np.transpose(in_matrix)] = 10

    return matrix

def move_box(label, traj, row_ind):
    clses,boxes = torch.tensor(label)[:,0].to(device),torch.tensor(label)[:,1:].to(device)
    xyxyboxes = xywh2xyxyn(boxes)
    traj_len = traj[0].shape[0]  
    result=[]  
    result.append((clses,boxes.clone()))
    for m in range(traj_len): #every frame in a window
        if m != 0:  #not the first frame, move

            for j, bind in enumerate(row_ind): #process every box
                if not if_in(xyxyboxes[bind],traj[j][0]):  #feature point not found
                    continue
                move_x = traj[j][m][0]-traj[j][m-1][0]
                move_y = traj[j][m][1]-traj[j][m-1][1]
                boxes[j][0] += move_x
                boxes[j][1] += move_y
            result.append((clses,boxes.clone()))

    return result


def move_resize(f_p, f_n, box):
    #f_p, f_n: [num_features,3]
    #box: xyxy
    #return: new box

    p_p, p_n = f_p[:,:2],f_n[:,:2]
    #cal the move
    d_x1 = min(p_n[:,0])-min(p_p[:,0])
    d_y1 = min(p_n[:,1])-min(p_p[:,1])
    d_x2 = max(p_n[:,0])-max(p_p[:,0])
    d_y2 = max(p_n[:,1])-max(p_p[:,1])

    nb = box.copy()
    nb[0],nb[1],nb[2],nb[3] = max(0,nb[0]+d_x1),max(0,nb[1]+d_y1),min(1,nb[2]+d_x2),min(1,nb[3]+d_y2)

    #print(nb)
    nb = xyxy2xywh(nb)
    return nb

def move_box_multi(label, traj, row_ind):
    clses,boxes = np.array(label)[:,0],np.array(label)[:,1:]
    traj_len = WINDOW 
    result=[]  
    result.append((clses,boxes.copy()))
    for t in traj:
        t[:,:,0],t[:,:,1]=t[:,:,0]/fw,t[:,:,1]/fh
    for m in range(traj_len): #every frame in a window
        
        if m != 0:  #not the first frame, move

            for j in range(len(traj)):
                if traj[j].shape[0]:

                    boxes[j] = move_resize(traj[j][:,m-1,:],traj[j][:,m,:],
                                            xywh2xyxy(boxes[j]))
            result.append((clses,boxes.copy()))

    return result

def rescale_box(box_xywh,area_xyxy):
    w = area_xyxy[2]-area_xyxy[0]
    h = area_xyxy[3]-area_xyxy[1]

    x1 = box_xywh[0]-area_xyxy[0]
    y1 = box_xywh[1]-area_xyxy[1]
    rescaled_xywh = np.array([x1/w,y1/h,box_xywh[2]/w,box_xywh[3]/h])
    #print(rescaled_xywh)
    t = xywh2xyxy(rescaled_xywh)
    #print(t)
    rescaled_xyxy = np.array([max(0,t[0]),max(0,t[1]),
                        min(1,t[2]),min(1,t[3])])
    #print(rescaled_xyxy)
    
    rescaled_xywh = xyxy2xywh(rescaled_xyxy)
    return rescaled_xywh

def rescale_box_2(box_xyxy,area_xyxy):
    w = area_xyxy[2]-area_xyxy[0]
    h = area_xyxy[3]-area_xyxy[1]
    
    x1 = box_xyxy[0]*w + area_xyxy[0]
    y1 = box_xyxy[1]*h + area_xyxy[1]
    x2 = box_xyxy[2]*w + area_xyxy[0]
    y2 = box_xyxy[3]*h + area_xyxy[1]

    rescaled_xyxy = np.array([max(0,x1),max(0,y1),min(1,x2),min(1,y2)])
    rescaled_xywh = xyxy2xywh(rescaled_xyxy)
    return rescaled_xywh

def framediff(img0,img1):
    k_size = 3
    threshold = 15

    es = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_size, k_size))

    if len(img0.shape) == 3: 
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(img0,img1)   

    #diff = cv2.medianBlur(diff, k_size)
    diff = cv2.GaussianBlur(diff, (k_size, k_size), 0)

    ret, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    mask = cv2.dilate(mask, es, 2) 
    mask = cv2.erode(mask, es, 2)  

    return mask#, cnts #diff, bounds


def if_legal(box, box_type=0):
    #box_type: 0-xywh, 1-xyxy
    if box[0]>0 and box[0]<1 and box[1]>0 and box[1]<1 and box[2]>0 and box[2]<1 and box[3]>0 and box[3]<1:
        if box_type:    #xyxy
            if box[2]>box[0] and box[3]>box[1]:         
                return True
        else:   #xywh
            if box[2]>0 and box[3]>0:
                return True
    return False

def save_txt(result, fid):
    isExist = os.path.exists(result_path)
    if not isExist:
        os.makedirs(result_path)

    clses,boxes = result
    for cls, box in zip(clses,boxes):
        cls,box = cls.tolist(),box.tolist()
        line = (cls,*box)
        with open(result_path+'{:0>6}.jpg'.format(fid), 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')
    return 0


def smooth_points(X, pos_thrs, dir_thrs, acc_thrs):
    P, N, D = X.shape
    V = np.gradient(X, axis=0)
    A = np.gradient(V, axis=0)
    x_diff = np.linalg.norm((X[-1]-X[0]), axis=-1)
  
    stationary_points = np.logical_or(np.less(x_diff, pos_thrs), np.greater(x_diff, 100))

    car_points = np.zeros(N, dtype=bool)
    
    angle = np.arctan2(V[:, :, 1], V[:, :, 0])
    direction = np.gradient(angle, axis=0)

    mask = np.all(np.abs(direction) < dir_thrs,axis=0)

    mask &= np.max(np.linalg.norm(A, axis=-1),axis=0) < acc_thrs
    

    car_points[mask & ~stationary_points] = True

    return np.nonzero(car_points)[0]


def normalize(data, mean, std):
    
    norm_ind = [0,1]
    mean, std = torch.tensor(mean), torch.tensor(std)
    data[:, :, norm_ind] = ((data[:, :, norm_ind] - mean) / std).float()

    return data

def denormalize(data, mean, std):

    norm_ind = [0,1]
    mean, std = torch.tensor(mean), torch.tensor(std)
    data[:, :, norm_ind] = ((data[:, :, norm_ind] *std) + mean).float()

    return data

def convert_cc_to_wc(points, P_inv):
    points[:,:,0] = points[:,:,0]*fw
    points[:,:,1] = points[:,:,1]*fh
    P_inv = torch.tensor(P_inv)
    z = torch.ones([points.shape[0],points.shape[1],1])
    cc = torch.cat([points, z], 2)

    wc = torch.zeros_like(points)
    for i, p in enumerate(points):
        wc_ = torch.matmul(P_inv.double(), cc[i].T.double()).T
        wc[i] = (wc_.T/(wc_[:,-1].unsqueeze(0).repeat(3,1))).T[:,[0,1]]
    return wc

def convert_wc_to_cc(points, P):

    P = torch.tensor(P)
    z = torch.ones([points.shape[0],points.shape[1],1])
    wc = torch.cat([points, z], 2)

    cc = torch.zeros_like(points)
    for i, p in enumerate(points):
        cc_ = torch.matmul(P.double(), wc[i].T.double()).T
        cc[i] = (cc_.T/(cc_[:,-1].unsqueeze(0).repeat(3,1))).T[:,[0,1]]
        
    cc[:,:,0] = cc[:,:,0]/fw
    cc[:,:,1] = cc[:,:,1]/fh
    return cc

def linear_partition(lst, k):
    n = len(lst)
    if k > n:
        k = n
    opt = np.zeros((n+1, k+1))
    d = np.zeros((n+1, k+1))

    for m in range(1, n+1):
        opt[m][1] = opt[m-1][1] + lst[m-1]

    for j in range(2, k+1):
        for m in range(1, j):
            opt[m][j] = opt[m][m]
            d[m][j] = d[m][j-1]

        for m in range(j, n+1):
            opt[m][j] = opt[m][j-1]
            d[m][j] = d[m][j-1]
            for i in range(1, m):
                cost = max(opt[i][j-1], opt[m][1] - opt[i][1])
                if opt[m][j] > cost:
                    opt[m][j] = cost
                    d[m][j] = i

    index_seg = []
    j = k
    m = n
    while j > 0:
        if m < j:
            j = m
        index_seg.insert(0, m)
        m = int(d[m][j])
        j -= 1

    return index_seg