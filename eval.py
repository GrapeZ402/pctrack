import torch
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str)
parser.add_argument('--gt_path', type=str)
args = parser.parse_args()

rp=args.result_path
lp=args.gt_path

thred = 0.45
pred_p=[]
count = 0

fid = 0

total_performance = np.array([0, 0, 0]).astype(float)


def box_iou(box1, box2):

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxyn(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


for filepath,dirnames,filenames in os.walk(rp):
    filenames.sort()
    for filename in filenames:
        label=[]
        pred=[]
        name_id = int(filename[:-4])

        with open(lp+filename, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')  
                label.append(list(float(i) for i in line.split()))
        with open(rp+filename, "r") as f:
            for line in f.readlines():
                line = line.strip('\n').split()
                pred.append(list(float(i) for i in line))
            
        pred_p=pred
        
        label = torch.tensor(label)
        pred = torch.tensor(pred)
        
        lbox = xywh2xyxyn(label[:,-4:])
        lcls = label[:, 0]
        pbox = xywh2xyxyn(pred[:,-4:])
        pcls = pred[:, 0]
        pcls[pcls==-1] = 2  

        iou = box_iou(lbox,pbox)
        nl,np_ = iou.shape[0],iou.shape[1]
        
        nc = [i for i in range(80)]
        for c in nc:
            TP = 0 
            FP = 0
            TPFN = 0
            tiou = 0

            pbox_, lbox_ = pbox[pcls==c], lbox[lcls==c]
            if len(pbox_)==0 and len(lbox_)==0:
                continue
            elif len(pbox_)==0 or len(lbox_)==0:
                FP = len(pbox_)
                TPFN = len(lbox_)
                precision = 0
                recall = 0
                F1 = 0
            
            else:
                iou = box_iou(pbox_, lbox_)
                used_l = np.zeros(len(lbox_))

                TPFN = len(lbox_)
                for i, iiou in enumerate(iou):
                    max_iou, max_iou_id = max(iiou), np.argmax(iiou)
                    if max_iou > thred and used_l[max_iou_id] == 0:
                        TP += 1
                        used_l[max_iou_id] = 1
                    else:
                        FP += 1
                precision = TP/(TP+FP) if TP+FP != 0 else 0
                recall = TP/(TPFN) if TPFN != 0 else 0
                F1 = 2*precision*recall/(precision+recall) if precision+recall!=0 else 0
            if fid%1 == 0:
                total_performance += np.array([precision, recall, F1])
                count += 1
        fid += 1

avg_prec, avg_recall, avg_f1 = total_performance[0]/count, total_performance[1]/count, total_performance[2]/count
print("precision: ",total_performance[0]/count, "recall: ",total_performance[1]/count, "F1 score: ",total_performance[2]/count)