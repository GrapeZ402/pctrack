import numpy as np
from utils.utils import *
import cv2
import time
import oapackage
from utils.config import correct_count


def simplify_sequence(seq, threshold=5):
    #minimize the length of list
    if len(seq) <= 15:
        return seq
    simplified = [seq[0]]
    for num in seq[1:]:
        if num - simplified[-1] >= threshold:
            simplified.append(num)
    return simplified

def get_cut_points_p(img):
    #input: grid, box !!!!
    def sobel_edge_detection(img):
        # Convert the image to grayscale
        #gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply the Sobel operator kernels to the image
        G_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        G_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        # Calculate the magnitude of the gradient
        P_y = cv2.magnitude(G_y, np.zeros_like(G_y))
        P_x = cv2.magnitude(G_x, np.zeros_like(G_x))
        # Normalize the gradient values
        P_y = P_y / np.max(P_y)*255
        P_x = P_x / np.max(P_x)*255
        # Return the gradient image
        return P_x, P_y
    
    # Apply the Sobel edge detection algorithm
    edge_x, edge_y = sobel_edge_detection(img)
    _, edge_x1 = cv2.threshold(edge_x, 254, 255, cv2.THRESH_BINARY)
    _, edge_y1 = cv2.threshold(edge_y, 254, 255, cv2.THRESH_BINARY)
    
    
    # Find the indices where the edge images equal 255
    x_indices = np.where(edge_x1 == 255)[1]
    y_indices = np.where(edge_y1 == 255)[0]
    
    # Remove duplicates and sort the indices
    xList = sorted(set(x_indices))
    yList = sorted(set(y_indices))
    #print(len(xList),len(yList))
    xList = simplify_sequence(xList, int(img.shape[0]/5))
    yList = simplify_sequence(yList, int(img.shape[1]/5))
    #print("list len:",len(xList),len(yList))
    return xList, yList

#优化目标
def diff_inside(grid, box):
    #grid: consist of 0 and 1
    #box: n x 4
    h,w = grid.shape[0],grid.shape[1]
    sum = np.zeros([len(box)])
    sum_all = grid.sum()
    for i,b in enumerate(box):
        sub_grid = grid[b[1]:b[3],b[0]:b[2]]
        sum[i] = sub_grid.sum()
    rate = sum/((box[:,3]-box[:,1])*(box[:,2]-box[:,0]))
    #return sum/(h*w), rate
    return sum/sum_all, rate    #sum记为框内总白点占区域总白点的比

def box_area(box):
    # box = 4xn
    return (box[2] - box[0]) * (box[3] - box[1])
    # box = 4xn
    #return np.prod(box[2:] - box[:2])

def iou_derection(box,box_pre):
    #box: xyxy n x 4
    #box_pre: 1 x 4
    box_pre = np.expand_dims(box_pre, 0)

    area1 = box_area(box.T)
    area2 = box_area(box_pre.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    #inter = (torch.min(box[2:], box_pre[2:]) - torch.max(box[:2], box_pre[:2])).clamp(0).prod(0)
    inter = (np.minimum(box[:, 2:], box_pre[:, None, 2:]) - np.maximum(box[:, :2], box_pre[:, None, :2])).clip(0).prod(2)
    return (inter / (area1 + area2[:, None] - inter))  # iou = inter / (area1 + area2 - inter)

def iou_derection1(box,box_pre):
    #box； xyxy tensor

    area1 = box_area(box.T)
    area2 = box_area(box_pre.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    #inter = (torch.min(box[2:], box_pre[2:]) - torch.max(box[:2], box_pre[:2])).clamp(0).prod(0)
    inter = (np.minimum(box[2:], box_pre[2:]) - np.maximum(box[:2], box_pre[:2])).clip(0).prod(0)
    return inter / (area1 + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def iou_surround(box, surr_box):
    if not len(surr_box):
        return np.zeros(len(box))
    #box = box.unsqueeze(0)
    
    area1 = box_area(surr_box.T)
    area2 = box_area(box.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    #inter = (torch.min(box[:, None, 2:], surr_box[:, 2:]) - torch.max(box[:, None, :2], surr_box[:, :2])).clamp(0).prod(2)
    inter = (np.minimum(box[:, None, 2:], surr_box[:, 2:]) - np.maximum(box[:, None, :2], surr_box[:, :2])).clip(0).prod(2)
    return (inter / area1).sum().item()  # iou = inter / (area1 + area2 - inter)

#pareto fixer
def generate_rectangles(x_list, y_list, distance=5):
    #生成可能的坐标框
    def combinations_with_distance(lst, distance):
        return [(lst[i], lst[j]) for i in range(len(lst)) for j in range(i+distance, len(lst))]

    x_combinations = combinations_with_distance(x_list, int(len(x_list)/2))
    y_combinations = combinations_with_distance(y_list, int(len(y_list)/2))
    rectangles = [(x[0], y[0], x[1], y[1]) for x in x_combinations for y in y_combinations]
    return rectangles

def pareto_fixer(box, grid, xlist, ylist, surr_box):
    xlist = np.array(xlist)
    ylist = np.array(ylist)

    h,w = grid.shape[0], grid.shape[1]
    x1,y1,x2,y2 = int(box[0]*w),int(box[1]*h),int(box[2]*w),int(box[3]*h)
    #print(x1,y1,x2,y2)
    if len(surr_box):
        surr_box = surr_box * np.array([w, h, w, h])
        #将surr_box覆盖区域置黑
        m = -2   #扩大/缩小一点
        mask0 = np.zeros_like(grid)
        mask0[y1+m:y2-m,x1+m:x2-m] = 1
        mask1 = np.ones_like(grid)
        for sb in surr_box:
            exist = sb.astype(int)
            mask1[exist[1]-m:exist[3]+m, exist[0]-m:exist[2]+m] = 0
        mask = np.logical_or(mask0,mask1)
        grid[mask==0] = 0

    x1_p = max(0,np.searchsorted(xlist, x1, side='right')-1)
    x2_p = max(0,np.searchsorted(xlist, x2, side='right')-1)
    y1_p = max(0,np.searchsorted(ylist, y1, side='right')-1)
    y2_p = max(0,np.searchsorted(ylist, y2, side='right')-1)

    anchors = np.array(generate_rectangles(xlist, ylist))
    #print(xlist,ylist,grid.shape)
    a_iou = box_iou_np(anchors, np.array([[x1,y1,x2,y2]]))

    anchors = anchors[np.squeeze(a_iou) > 0.2,:]
    
    sum_n, rate_n = diff_inside(grid, anchors)
    iou_n = iou_derection(anchors, np.array([x1, y1, x2, y2])).squeeze(0)
    iou_surr_n = iou_surround(anchors, surr_box)

    sum_n = sum_n[iou_n!=1]
    rate_n = rate_n[iou_n!=1]
    # iou_surr_n = iou_surr_n[iou_n!=1]
    iou_n = iou_n[iou_n!=1]
    
    
    pareto=oapackage.ParetoDoubleLong()
    for ii in range(0, len(sum_n)):
        ww = oapackage.doubleVector( (sum_n[ii], rate_n[ii], iou_n[ii])) # S, -iou_surr_n[ii]))
        pareto.addvalue(ww, ii)
    lst=np.array(pareto.allindices())

    #print(np.argmax(sum_n[lst]+rate_n[lst]+iou_n[lst]))
    #print(lst)
    opt_ind = lst[np.argmax(sum_n[lst]+0.3*rate_n[lst]+0.3*iou_n[lst])]
    #print(sum_n[opt_ind],rate_n[opt_ind],iou_n[opt_ind])
    optimals = anchors[opt_ind]
    optimals = optimals.T
    
    return [optimals[0]/w, optimals[1] / h,optimals[2] / w, optimals[3] / h]

#fix function
def fix_box(frame_p, frame, frame_n, box_xywh, pred_xywh):
    global correct_count
    #cal the downsampled diff
    #box_xywh: (numpy array) box to be fixed
    #pred_xyxy: all detect boxes
    left = 0    #是否离开了画面
    t1 = time.time()
    scale = 2
    downsample = 5

    pred_xyxy = xywh2xyxyn(pred_xywh)

    #transform the position
    area_xywh = box_xywh.copy()
    area_xywh[2], area_xywh[3] = area_xywh[2]*scale, area_xywh[3]*scale
    area_xyxy = cut_outside(xywh2xyxy(area_xywh))

    r_box_xywh = rescale_box(box_xywh, area_xyxy)
    r_box_xyxy = xywh2xyxy(r_box_xywh)

    #ta = frame_n[int(area_xyxy[1]*fh):int(area_xyxy[3]*fh),int(area_xyxy[0]*fw):int(area_xyxy[2]*fw),:]
    #cv2.imwrite('test_area.jpg',ta)

    h, w = frame.shape[0], frame.shape[1]   #original h w
    hd, wd = int(h/downsample), int(w/downsample) #downsampled h w (384,216)
    frame = cv2.resize(frame, [int(w/downsample),int(h/downsample)])
    frame_p = cv2.resize(frame_p, [int(w/downsample),int(h/downsample)])
    frame_n = cv2.resize(frame_n, [int(w/downsample),int(h/downsample)])

    #test diff
    """
    path = 'D:/code/trackbypredict/results/data1/diff-plot/'
    z = '000000'
    diff_full = framediff(frame_p,frame_n)
    cv2.imwrite(path+ z[0:-len(str(fid))]+str(fid) + '.jpg', diff_full)
    """


    frame = frame[int(area_xyxy[1]*hd):int(area_xyxy[3]*hd),
                    int(area_xyxy[0]*wd):int(area_xyxy[2]*wd)]
    frame_p = frame_p[int(area_xyxy[1]*hd):int(area_xyxy[3]*hd),
                    int(area_xyxy[0]*wd):int(area_xyxy[2]*wd)]
    frame_n = frame_n[int(area_xyxy[1]*hd):int(area_xyxy[3]*hd),
                    int(area_xyxy[0]*wd):int(area_xyxy[2]*wd)]
    
    try:
        diff = framediff(frame,frame_n)   #data1用(frame_p,frame_n)，其他用(frame_p,frame)
        #diff = framediff3(frame_p,frame,frame_n)
        _, diff_down = cv2.threshold(diff, 128, 255, cv2.THRESH_BINARY)
        grid = diff_down
        
        m = np.zeros_like(grid)
        m[grid > 128] = 1
        count = np.sum(m)
        #grid = diff_down[int(area_xyxy[1]*hd):int(area_xyxy[3]*hd),
        #                int(area_xyxy[0]*wd):int(area_xyxy[2]*wd)]
        #cv2.imwrite("data3_diff.jpg",diff_down)
    
        #周围存在其他物体
        surr_box = []
        iou = box_iou_np(np.expand_dims(area_xyxy,0), pred_xyxy)[0]
        for i, iiou in enumerate(iou): 
            if iiou != 0 and iiou < 0.249:
                surr_box.append(np.expand_dims(xywh2xyxy(rescale_box(pred_xywh[i], area_xyxy)),0))
        if len(surr_box):surr_box = np.concatenate(surr_box,0)
        #print(surr_box)
        t2 = time.time()
        #print("预处理1:",t11-t1,"预处理2:",t12-t11,"预处理3:",t13-t12,"预处理4:",t14-t13,"预处理5:",t2-t14)
        #print("预处理: ",t2 - t1)
    
        
        xlist, ylist = get_cut_points_p(grid)
        t3= time.time()
        #print("角点：",t3 - t2)
        #search
    
        #判定离开
        if count < 0.05*(grid.shape[0]*grid.shape[1]):   #帧差太少, 离开或没动
            if if_in(left_area[0],(box_xywh[0],box_xywh[1])):
                left = 1
                return box_xywh, left
            return box_xywh, left
        
        t3 = time.time()
        #print("??", box_xywh, count, (grid.shape[0]*grid.shape[1]))
        result_xyxy = pareto_fixer(r_box_xyxy, m, xlist, ylist, surr_box)
        #print(result_xyxy)
        #result_xywh = xyxy2xywh(result_xyxy)
        
        #=========可视化一下=======
        """
        dh, dw = diff.shape[0], diff.shape[1]
        diff = cv2.rectangle(diff, (int(r_box_xyxy[0]*dw), int(r_box_xyxy[1]*dh)), (int(r_box_xyxy[2]*dw), int(r_box_xyxy[3]*dh)), (80,255,0),1)
        diff = cv2.rectangle(diff, (int(result_xyxy[0]*dw), int(result_xyxy[1]*dh)), (int(result_xyxy[2]*dw), int(result_xyxy[3]*dh)), (160,255,0),1)
        cv2.imwrite("/home/zhangxinyi/zxy/scripts/plots/data2_corrector/"+str(correct_count)+'.jpg', diff)
        correct_count+= 1
        """
        
        
        result_xywh = rescale_box_2(result_xyxy,area_xyxy)
        t4 = time.time()
    
        #print("搜索耗时：",t4 - t3)
        return result_xywh, left
    except:
        return box_xywh, left