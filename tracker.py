import cv2
from utils.utils import *
from utils.config import *
from corrector import *
import torch
import time
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from model.cnn import CNNPred
import numba


class frame_cache:
    def __init__(self) -> None:
        self.frames = []
        self.diff = []
        self.start_id = 0

    def fill_full(self, id):
        #从id开始读取2*window帧存入cache, gray
        
        self.diff = []
        self.frames = []
        
        self.start_id = id-WINDOW   #缓存的是当前帧的前一个窗口加当前窗口
        z = '000000'
        for i in range(self.start_id, self.start_id+2*WINDOW+1):    
  
            img = cv2.imread(source + '{:0>6}.jpg'.format(i))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.frames.append(img)
        
        """
        if not len(self.frames):
            self.start_id = id-WINDOW   #缓存的是当前帧的前一个窗口加当前窗口
            z = '000000'
            for i in range(self.start_id, self.start_id+2*WINDOW+1):    
      
                img = cv2.imread(img_path + z[0:-len(str(i))] + str(i) + '.jpg')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.frames.append(img)
                
        else:
            self.frames = self.frames[WINDOW:-1]
            self.start_id = id
            z = '000000'
            for i in range(self.start_id, self.start_id+WINDOW+1):    
      
                img = cv2.imread(img_path + z[0:-len(str(i))] + str(i) + '.jpg')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.frames.append(img)
        """      
                
        return 0
    
    #@profile
    def glimpse_choose(self):
        diff_list = []
        thres = 35  #glimpse给的阈值
        #计算每一帧帧差
        for i in range(1,WINDOW):
            frame0 = self.frames[i]
            frame1 = self.frames[i+1]
            diff = cv2.absdiff(frame0, frame1)
            #self.diff.append(diff)
            _, mask = cv2.threshold(diff, thres, 1, cv2.THRESH_BINARY)
            diff_list.append(np.sum(mask))
        #choose frames
        k = int(np.ceil(0.1*WINDOW)) #glimpse给的比例
        #print(k)
        track_list = linear_partition(np.array(diff_list),k)

        return track_list

    def get_one_frame(self,id):
    
        cid = id - self.start_id
        #print(cid,len(self.frames))
        frame = self.frames[cid]
        return frame
    

class Tracker:
    def __init__(self, label, start_id, method):
        self.label = np.array(label)
        self.method = method
        
        self.cache = frame_cache() #先将两个窗口的视频帧缓存到内存中
        self.cache.fill_full(start_id)

        self.first_gray = self.cache.frames[0]

        self.prev_id = start_id
        self.curr_id = start_id + 1

        self.prev_gray = self.cache.frames[WINDOW]
        self.curr_gray = self.cache.frames[WINDOW+1]

        self.prev_pts = np.array([], dtype=np.float32)
        self.curr_pts = np.array([], dtype=np.float32)

        self.features = np.array([], dtype=np.float32)
        self.features_no = np.array([], dtype=np.float32)   #存储新目标的特征点

        self.prev_box = np.array(self.label)[:,1:]
        self.curr_box = np.array(self.label)[:,1:]
        self.pprev_box = np.array(self.label)[:,1:]

        self.cls = self.label[:,0]
        self.prev_cls = self.label[:,0]
        self.pprev_cls = self.label[:,0]

        self.newpType = np.array([], dtype=np.float32)   #记录属于新目标的特征点在features中的下标
        self.new_obj_count = 0
        self.NeededFix = np.zeros(len(self.curr_box))
        self.NeededPred = np.zeros(len(self.curr_box))
        self.isNewBox = np.zeros(len(self.curr_box))

        self.jumped = True     #记录上一次操作是否是跳帧
        self.speed = np.zeros([len(self.curr_box),2])   #每个目标框的移动速度
        #self.vars = np.zeros(len(self.curr_box))    #特征点移动方差

        #self.pred_model = LSTM(LSTMParams)
        if method == 'ours':
            self.pred_model = CNNPred(in_channels=5,hidden_size=128,output_size=20).to(device)
            checkpoint = torch.load(weight_path, map_location=torch.device(device))  # 加载checkpoint并指定map_location为cpu
            self.pred_model.load_state_dict(checkpoint['model'])  # 加载模型参数

        self.pred_input = torch.cat([torch.tensor(self.curr_box),torch.tensor(self.cls).unsqueeze(1)],1).unsqueeze(0)
        self.pred_output = torch.tensor([])
        self.pred_mask = []

        #self.gt = []

    
    #@profile
    def reset(self, new_label, start_id):
        #reset the label and the first frame
        t1 = time.time()

        self.cache.fill_full(start_id)

        self.label = new_label
        self.prev_id = start_id
        self.curr_id = start_id + 1
        #self.prev_box = self.curr_box
        self.pprev_box = self.curr_box
        #self.curr_box = np.array(self.label)[:,1:]    #改
        self.prev_box = np.array(self.label)[:,1:]
        self.prev_cls = self.cls
        self.cls = np.array(self.label)[:,0]
        self.first_gray = self.cache.frames[0]

        t2 = time.time()
        return 0
    
    def get_gray_img(self,id):
        z = '000000'
        #img = cv2.imread(img_path + z[0:-len(str(id))] + str(id) + '.jpg')
        img = self.cache.get_one_frame(id)

        return img
    
    #@profile
    def glimpse_IDP(self):

        idp_id = self.cache.glimpse_choose()
        idp_id = np.insert(idp_id,0,0)

        p0 = self.curr_pts
        features = np.expand_dims(p0.squeeze(1),0)
        for i in range(len(idp_id)-1):
            frame0, frame1 = self.cache.frames[idp_id[i]], self.cache.frames[idp_id[i+1]]
            p1,st,_ = cv2.calcOpticalFlowPyrLK(frame0, frame1, p0, None,winSize=(15,15),maxLevel=10)
            best_new = p1[st==1]
            p0 = best_new.reshape(-1,1,2)

            features = np.array([f.reshape(-1,1,2)[st==1] for f in features])
            features = np.concatenate([features, np.expand_dims(best_new,0)],0)
        
        #move the bounding box
        c_box = self.prev_box.copy()
        #self.prev_box = self.curr_box.copy()
        p_pts, c_pts = features[0], features[-1]
        features = np.transpose(features,[1,0,2])
        matrix = creat_matrix(c_box, features, fw,fh)

        for i, m in enumerate(matrix):
            ind = np.where(m == 10)[0]   #简单选点，需改

            if len(ind):    #有可能选不到点
                pp, cp = p_pts[ind]/np.array([[fw,fh]]), c_pts[ind]/np.array([[fw,fh]])     #c_pts和p_pts应该一一对应
                #print(pp, cp, c_box[i])
                c_box[i] = move_resize(pp, cp, xywh2xyxy(c_box[i]))
        self.prev_box = c_box.copy()

        #set
        #print("prev id",self.prev_id,"curr id",self.curr_id)
        self.prev_gray = self.get_gray_img(self.prev_id)
        self.curr_gray = self.get_gray_img(self.curr_id)
        self.first_gray = self.curr_gray.copy()
        
        return 0
    
    def no_IDP(self):
        self.prev_gray = self.get_gray_img(self.prev_id)
        self.curr_gray = self.get_gray_img(self.curr_id)
        self.first_gray = self.curr_gray.copy()
        return 0
    
    def glimpse_IDP_test(self):

        idp_id = self.cache.glimpse_choose()
        idp_id = np.insert(idp_id,0,0)
        res = []
        p0 = self.curr_pts
        features = np.expand_dims(p0.squeeze(1),0)
        for i in range(len(idp_id)-1):
            frame0, frame1 = self.cache.frames[idp_id[i]], self.cache.frames[idp_id[i+1]]
            p1,st,_ = cv2.calcOpticalFlowPyrLK(frame0, frame1, p0, None,winSize=(15,15),maxLevel=10)
            best_new = p1[st==1]
            p0 = best_new.reshape(-1,1,2)

            features = np.array([f.reshape(-1,1,2)[st==1] for f in features])
            features = np.concatenate([features, np.expand_dims(best_new,0)],0)
        
        #move the bounding box
        c_box = self.curr_box.copy()
        res.append(c_box.copy())
        #p_pts, c_pts = features[0], features[-1]
        features = np.transpose(features,[1,0,2])
        matrix = creat_matrix(c_box, features, fw,fh)
        #print(features.shape)

        for j in range(features.shape[1]-1):
            p_pts, c_pts = features[:,j,:], features[:,j+1,:]
            for i, m in enumerate(matrix):
                ind = np.where(m == 10)[0]   #简单选点，需改

                if len(ind):    #有可能选不到点
                    pp, cp = p_pts[ind]/np.array([[fw,fh]]), c_pts[ind]/np.array([[fw,fh]])     #c_pts和p_pts应该一一对应
                    #print(pp, cp, c_box[i])
                    c_box[i] = move_resize(pp, cp, xywh2xyxy(c_box[i]))
            res.append(c_box.copy())
        self.prev_box = c_box.copy()

        #set
        #print("prev id",self.prev_id,"curr id",self.curr_id)
        self.prev_gray = self.get_gray_img(self.prev_id)
        self.curr_gray = self.get_gray_img(self.curr_id)
        self.first_gray = self.curr_gray.copy()
        
        return res, idp_id
    
    #@profile
    def predict_IDP(self):

        pred_id = np.array([0,5,10])

        p0 = self.curr_pts

        features = np.expand_dims(p0.squeeze(1),0)
        for i in range(len(pred_id)-1):
            frame0, frame1 = self.cache.frames[pred_id[i]], self.cache.frames[pred_id[i+1]]
            p1,st,_ = cv2.calcOpticalFlowPyrLK(frame0, frame1, p0, None,winSize=(15,15),maxLevel=10)
            best_new = p1[st==1]
            p0 = best_new.reshape(-1,1,2)

            features = np.array([f.reshape(-1,1,2)[st==1] for f in features])
            features = np.concatenate([features, np.expand_dims(best_new,0)],0)
        
        #move the bounding box(choose features)
        """
        c_box = self.curr_box.copy()
        features = np.transpose(features,[1,0,2])
        matrix = creat_matrix(c_box, features, fw,fh)

        input = np.expand_dims(c_box.copy(),0)
        for j in range(features.shape[1]-1):    #for every frame(3)
            p_pts, c_pts = features[:,j,:], features[:,j+1,:]
            for i, m in enumerate(matrix):
                ind = np.where(m == 10)[0]   #简单选点，需改

                if len(ind):    #有可能选不到点
                    pp, cp = p_pts[ind]/np.array([[fw,fh]]), c_pts[ind]/np.array([[fw,fh]])     #c_pts和p_pts应该一一对应
                    #print(pp, cp, c_box[i])
                    c_box[i] = move_resize(pp, cp, xywh2xyxy(c_box[i]))
            input = np.concatenate([input,np.expand_dims(c_box.copy(),0)],0)

        """
        #print("预测前", self.prev_box)
        c_box = self.prev_box.copy()    
        features = np.transpose(features,[1,0,2])
        row_ind, col_ind, vars = choose_tracklet(c_box, self.speed, features,4)

        self.NeededFix = vars>params['maxVars']

        input = np.expand_dims(c_box.copy(),0)
        for j in range(features.shape[1]-1):    #for every frame(3)
            p_pts, c_pts = features[:,j,:], features[:,j+1,:]
            for i, ind in enumerate(col_ind):   #for every box
                ind = ind[ind!=-1]

                if len(ind):    #有可能选不到点
                    pp, cp = p_pts[ind]/np.array([[fw,fh]]), c_pts[ind]/np.array([[fw,fh]])     #c_pts和p_pts应该一一对应
                    c_box[i] = move_resize(pp, cp, xywh2xyxy(c_box[i]))
  
            input = np.concatenate([input,np.expand_dims(c_box.copy(),0)],0)
        

        #predict
        c = np.repeat(np.reshape(self.cls,(1,len(self.cls),1)),input.shape[0],axis=0)
        input = np.concatenate([input,c], 2)

        input = torch.tensor(input).permute(1,0,2).float().to(device)
        #print(input)
        input[:,:,[0,1]] = convert_cc_to_wc(input[:,:,[0,1]], params['P_inv'])
        #print(input)
        input = normalize(input, params['mean'], params['std'])
        #print(input)
        output = self.pred_model(input)
        #print(output)
        output = denormalize(output, params['mean'], params['std'])
        output[:,:,[0,1]] = convert_wc_to_cc(output[:,:,[0,1]], params['P'])

        output = np.array(output.detach())
        
        k = int((WINDOW-25)/5)
        self.prev_box = output[:,k,:]
        #print("预测后",self.prev_box)
        #print(self.curr_box)
        
        #print("prev id",self.prev_id,"curr id",self.curr_id)
        self.prev_gray = self.get_gray_img(self.prev_id)
        self.curr_gray = self.get_gray_img(self.curr_id)
        self.first_gray = self.curr_gray.copy()
    

        #继承上一个窗口的新目标

        new_b = self.pprev_box[self.isNewBox==1]
        new_c = self.prev_cls[self.isNewBox==1] 
        if 1 in self.isNewBox:
            
            newb_iou = box_iou_np(xywh2xyxyn(new_b), xywh2xyxyn(self.prev_box))
            ind = np.array([i for i,iiou in enumerate(newb_iou) if max(iiou) < 0.3])
            if len(ind):
                new_b = new_b[ind]
                new_c = new_c[ind]
                self.prev_box = np.concatenate([self.prev_box,new_b],0)
                self.cls = np.concatenate([self.cls,new_c],0)

        self.NeededFix = np.zeros(len(self.prev_box))
        self.NeededPred = np.zeros(len(self.prev_box))
        self.isNewBox = np.zeros(len(self.prev_box))
        self.speed = np.zeros([len(self.prev_box),2])

        #fixer
        #self.NeededFix[self.curr_box[:,1]>0.3] = 1

        # _ = self.diffixer()
        return 0
    
    def update_after_dp(self):
        self.curr_box = self.prev_box
        return 0
    
    def get_features(self):
        return np.transpose(np.array(self.features),(1,0,2))
    
    def get_pts(self):
        return self.prev_pts.squeeze(1), self.curr_pts.squeeze(1)
    
    def getBboxResult(self):
        return (self.cls, self.curr_box.copy())
    
    def detectFeatures(self, quality = feature_quality_level):

        t1 = time.time()

        mask_use = creat_mask(self.first_gray, self.prev_box, fw, fh)  
        harris = True
        if data_param == 2:
            harris = False    
        p0 = cv2.goodFeaturesToTrack(self.first_gray, mask=mask_use, maxCorners=500, qualityLevel=quality, minDistance=7, useHarrisDetector=harris)

        self.features = np.expand_dims(p0.squeeze(1),0)
        self.prev_pts = p0
        self.curr_pts = p0

        self.newpType = np.zeros(len(self.curr_pts))    #用来记录特征点的类型

        t2 = time.time()
        return 0
    
    def OpticalFlow(self):  

        t1 = time.time()
        self.jumped = False

        p1,st,_ = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.curr_gray, self.curr_pts, None,winSize=(15,15),maxLevel=10)

        self.prev_gray = self.curr_gray.copy()
        self.curr_gray = self.get_gray_img(self.curr_id)

        #st_0 = st[self.new_obj_ind==0].reshape(-1,1)
        #self.features = np.array([f.reshape(-1,1,2)[st_0==1] for f in self.features])
        #self.features = np.concatenate([self.features, np.expand_dims(p1[self.new_obj_ind==0][st_0==1],0)], 0)
        #保存与新目标无关的特征点，并作筛选与匹配（新目标特征点永远属于该新目标无需匹配）
        best_new = p1[st==1]
        self.prev_pts = self.curr_pts[st==1].reshape(-1,1,2)
        self.curr_pts = best_new.reshape(-1,1,2)
        
        #print(self.features.shape,len(st),self.new_obj_ind.shape,np.expand_dims(best_new.reshape(-1,1,2)[st_0==1],0).shape)

        self.prev_id = self.curr_id
        self.curr_id = self.curr_id + 1

        #新目标特征点    
        if np.any(self.newpType):   #当前存在新目标特征点
            self.new_obj_count += 1
            st_1 = st[self.newpType!=0].reshape(-1,1)
            self.features_no = np.array([f.reshape(-1,1,2)[st_1==1] for f in self.features_no])
            self.features_no = np.concatenate([self.features_no, np.expand_dims(p1[self.newpType!=0][st_1==1],0)], 0)
        self.newpType = self.newpType.reshape(-1,1)[st==1]
        #print(self.features_no.shape)
        t2 = time.time()

        return 0
    
    def jump_frame(self):

        #1 frame
        self.curr_gray = self.get_gray_img(self.curr_id)

        self.prev_id = self.curr_id
        self.curr_id = self.curr_id + 1
        #print("prev id",self.prev_id,"curr id",self.curr_id)
        #self.prev_box = self.curr_box
        self.prev_pts = self.curr_pts

        self.jumped = True
        return 0
    
    def move_box(self):
        t1 = time.time()
        #jump的情况
        if self.jumped:
            t2 = time.time()
            return t2-t1
        #根据上一组特征点和本组特征点把当前框向后移动一次
        #执行顺序应在opticalflow后面

        #特征点如何筛选？（维护一个direction和acceleration）
        p_box = self.prev_box.copy()
        c_box = self.curr_box.copy()
        #cal speed of every curr box
        cp_iou = box_iou_np(xywh2xyxyn(c_box),xywh2xyxyn(p_box))

        if not self.jumped:
            """
            for ic, iou in enumerate(cp_iou):
                if max(iou)>0.2:
                    ip = np.argmax(iou)
                    self.speed[ic] = c_box[ic][[0,1]] - p_box[ip][[0,1]]
                else:
                    self.speed[ic] = np.array([-1,-1])
            """
            self.prev_box = self.curr_box.copy()        
        p_pts, c_pts = self.get_pts()
        
        #if 1:
        if self.method == "base":
            #----不特征点选择----
            matrix = creat_matrix(c_box, np.concatenate((self.prev_pts,self.curr_pts),1), fw,fh)
    
            for i, m in enumerate(matrix):
                ind = np.where(m == 10)[0]   #简单选点，需改
    
                if len(ind):    #有可能选不到点
                    pp, cp = p_pts[ind]/np.array([[fw,fh]]), c_pts[ind]/np.array([[fw,fh]])     #c_pts和p_pts应该一一对应
                    #print(pp, cp, c_box[i])
                    c_box[i] = move_resize(pp, cp, xywh2xyxy(c_box[i]))
                    
        #if 0:
        if self.method == "ours": 
            #----特征点选择----
            #t1 = time.time()
            row_ind, col_ind, vars = choose_tracklet(c_box, self.speed, np.concatenate((self.prev_pts,self.curr_pts),1),4)
    
            #self.NeededFix = np.logical_or((vars>maxVars), self.NeededFix)
            self.NeededFix = vars>params['maxVars']
            #print(self.NeededFix)
            #self.NeededFix = np.where((vars>maxVars) | (self.NeededFix == 2), 1, self.NeededFix)
            #print(time.time()-t1)
       
            for i, ind in enumerate(col_ind):
                #ind = np.where(m == 10)[0]   #简单选点，需改
                ind = ind[ind!=-1]
    
                if len(ind):    #有可能选不到点
                    pp, cp = p_pts[ind]/np.array([[fw,fh]]), c_pts[ind]/np.array([[fw,fh]])     #c_pts和p_pts应该一一对应
                    #print(pp, cp, c_box[i])
    
                    c_box[i] = move_resize(pp, cp, xywh2xyxy(c_box[i]))
                    
                    #移动量
                    #if len(self.pred_mask):
                    #    if not self.pred_mask[i]:
                    #        c_box[i] = move_resize(pp, cp, xywh2xyxy(c_box[i]))
                    #else:
                    #    c_box[i] = move_resize(pp, cp, xywh2xyxy(c_box[i]))
                    
            #------------------
        

        #加入预测结果
        #print(self.pred_output.shape)
        if 0 not in self.pred_output.shape and self.NeededPred.any():   #需改
            #print(self.pred_mask.shape, self.NeededPred.shape)
            mask =  np.logical_and(self.pred_mask, self.NeededPred).astype(bool)
            #print(self.pred_mask)
            #try:    #需改
            #print("预测：", c_box[mask], self.pred_output[self.NeededPred[self.pred_mask]][:,2,:])
            #c_box[mask] = self.pred_output[mask][:,self.curr_id%WINDOW-WINDOW/2,:] #直接复制 or ＋移动量？
            c_box[mask] = self.pred_output[self.NeededPred[self.pred_mask]][:,-1,:] #直接复制 or ＋移动量？
            #self.pred_output = np.delete(self.pred_output, 0, axis=1)
            #except:
            #    pass


        self.curr_box = c_box.copy()
        
        #加入新目标的box-----------------   
        if self.new_obj_count == 3:
            """
            smooth_p = smooth_points(self.features_no, 0.3, 0.3, 1)

            #彻底删除非平滑点, 同时修改newpType
            st = np.zeros(self.features_no.shape[1], dtype=int)
            st[smooth_p] = 1 
            self.features_no = self.features_no[:,st==1,:]
            st = np.concatenate([np.ones(len(self.newpType[self.newpType==0])),st])
            self.newpType = self.newpType[st==1]
            self.curr_pts = self.curr_pts[st==1]

            mask = self.newpType[self.newpType!=0].astype(int)
            """
            tt = time.time()
            s = set(self.newpType)
            s.discard(0)
            #s.discard(1)
            for j, t in enumerate(s):  #对于每一个新目标区域
                #t是新目标分区编号
                mask = np.where(self.newpType[self.newpType!=0]==t)[0]
                #print(mask)
                X = self.features_no[:,mask,:]
                #print(t,j,mask.shape,X.shape,smooth_th[int(t)-1])
                smooth_p = smooth_points(X, params['smooth_th'][int(t)-1][0], params['smooth_th'][int(t)-1][1], params['smooth_th'][int(t)-1][2])
                #彻底删除非平滑点, 同时修改newpType
                #print(smooth_p,mask)
                st = np.ones(self.features_no.shape[1], dtype=int)
                
                st[mask] = 0
                st[mask[smooth_p]] = 1 
                #print(st,mask)
                self.features_no = self.features_no[:,st==1,:]
                
                
                st = np.concatenate([np.ones(len(self.newpType[self.newpType==0])),st])
                #print("stshape",st.shape,st)
                #print("npshape",self.newpType.shape,self.newpType)
                self.newpType = self.newpType[st==1]
                self.curr_pts = self.curr_pts[st==1]
                #print("shape1",self.newpType.shape)
                
                X = X[:,smooth_p]

                data = X[0]   #以第一帧位置聚类
                #data = np.concatenate(((X[1:,:] - X[:3,:]),np.expand_dims(X[0],0)),0).transpose(1,0,2)
                #data = data.reshape(-1,8)
                
                
                
                if not len(data):   #不存在平滑点
                    continue
                data = StandardScaler().fit_transform(data)

                dbscan = DBSCAN(eps=params['db_th'][int(t)-1][0], min_samples=params['db_th'][int(t)-1][1]).fit(data) #对应的参数不同

                c_labels = dbscan.labels_
                n_clusters_ = len(set(c_labels)) - (1 if -1 in c_labels else 0)

                st = np.ones(self.features_no.shape[1], dtype=int)  #用于去除冗余
                #print(n_clusters_)
                for i in range(n_clusters_):
                    cluster = np.where(c_labels == i)[0]
                    ps = X[:,cluster,:][-1]
                    
                    top = ps[:,1].min()
                    bottom = ps[:,1].max()
                    left = ps[:,0].min()
                    right = ps[:,0].max()
           
                    a, b = bottom-top, right-left
                    S = a*b
                    R = b/a
                    
                    #print("!~~~!!!",S,R)
                    if np.abs(S-params['s_range'][int(t)-1]) > 0.5*params['s_range'][int(t)-1] or np.abs(R-params['r_range'][int(t)-1]) > 0.5*params['r_range'][int(t)-1]:
                    #if S<200 or R<r_min[int(t)-1] or S>s_max[int(t)-1] or R>2*r_max[int(t)-1]:
                        #print("111")
                        continue
                    
                    n_bbox = xyxy2xywh(np.array([left,top,right,bottom])/np.array([fw,fh,fw,fh]))
                    #避免重复框 需改
                    iou = box_iou_np(np.expand_dims(xywh2xyxy(n_bbox),0),xywh2xyxyn(self.curr_box))
                    if max(iou[0]) > 0.3:
                        continue
                    #print(max(iou[0]))
                    self.curr_box = np.concatenate([self.curr_box,np.expand_dims(n_bbox,0)],0)
                    self.cls = np.append(self.cls, -1)
                    print("新目标: ",n_bbox)
                    if t==2:    #大一点的区域
                        self.NeededFix = np.append(self.NeededFix, 2)   #新加入时设为需修正
                    else:   #小目标区域不修正
                        self.NeededFix = np.append(self.NeededFix, 0)

                    self.NeededPred = np.append(self.NeededPred, 0) #新加入时设为无需预测
                    self.isNewBox = np.append(self.isNewBox, 1)

                    #去除冗余特征点，随机选得了
                    n_pts = len(cluster)
                    if n_pts > 100:     #每个目标最多保留100个
                        random_ind = np.random.choice(cluster, len(cluster)-99, replace=False)
                        
                        #print(random_ind.shape, mask.shape, cluster.shape, st.shape,self.features_no.shape)
                        st[mask[random_ind]] = 0
              
                self.features_no = self.features_no[:,st==1,:]
                st = np.concatenate([np.ones(len(self.newpType[self.newpType==0])),st])

                self.newpType = self.newpType[st==1]
                self.curr_pts = self.curr_pts[st==1]
            print("新目标！！！",time.time()-tt)
        t2 = time.time()

        return 0

    # @profile
    def det_newobj(self, newobj_area):

        pred_xyxy = xywh2xyxyn(self.curr_box)
        new_p = self.det_newobj_features(newobj_area,pred_xyxy)

        self.newpType = np.zeros(len(self.curr_pts))
        for i, nps in enumerate(new_p):   #对于每一个新目标区域
            if len(nps):
                self.new_obj_count = 0
                self.newpType = np.concatenate((self.newpType, np.full(len(nps),i+1)),0)
                self.curr_pts = np.concatenate((self.curr_pts,np.expand_dims(nps,1)),0)
        if np.any(self.newpType):
            self.features_no = np.expand_dims(self.curr_pts[self.newpType!=0].squeeze(1),0)
        return 0

    def det_newobj_features(self, newobj_area, pred_xyxy):
        #运行应在move_box之后
        
        #cv2.imwrite("test_img1.jpg",self.curr_gray)
        #cv2.imwrite("test_img2.jpg",self.prev_gray)
        #print(self.curr_id,self.prev_id)
        hd, wd = int(fh/downsample), int(fw/downsample) #downsampled h w (384,216)
        #frame = cv2.resize(self.curr_gray.copy(), [wd,hd])
        #frame_p = cv2.resize(self.prev_gray.copy(), [wd,hd])
        frame = cv2.resize(self.get_gray_img(self.curr_id), [wd,hd])
        frame_p = cv2.resize(self.get_gray_img(self.prev_id), [wd,hd])
        
        diff = framediff(frame, frame_p)
        
        result = []
        newobj_iou = box_iou_np(newobj_area, pred_xyxy)
        for ino, iiou in enumerate(newobj_iou):

            thrs = params['diff_th'][ino]
            exist_ind = np.where(iiou)

            if exist_ind:   #当前区域存在已知目标，设为0
                for ex in exist_ind[0]:
                    m = 2   #扩大一点
                    exist = (pred_xyxy[ex]*np.array([wd,hd,wd,hd])).astype(int)
                    diff[exist[1]-m:exist[3]+m, exist[0]-m:exist[2]+m] = 0

            bd = (newobj_area[ino]*np.array([wd,hd,wd,hd])).astype(int)
            b = (newobj_area[ino]*np.array([fw,fh,fw,fh])).astype(int)
            diff_new = diff[bd[1]:bd[3],bd[0]:bd[2]]
        
            s = diff_new.sum() / 255
            #print("新目标",ino,s)
            #用于加入到光流特征点中的光流点
            if s > thrs:
                
                #加一个边缘检测
                #cut_p = get_cut_points(diff_new)
                cut_p = np.column_stack(np.where(diff_new == 255))
                coords = cut_p*np.array([downsample,downsample])+[b[1],b[0]]
                coords[:,[0,1]] = coords[:,[1,0]]
                result.append(np.array(coords,np.float32))

            else:
                result.append([])
        
        return result
    
    # @profile
    def diffixer(self):
        t1 = time.time()

        #运行逻辑应在move box之后
        #新目标仅在最开始修一次，需改
        
        #print("curr_id, prev_id",self.curr_id, self.prev_id)
        boxes0, boxes = np.array(self.label)[:,1:], self.curr_box  #boxes0是窗口首帧，需改，选出不为新目标的
        boxes_p = self.prev_box
        img0_d, img1 = self.prev_gray, self.curr_gray  #错的
        img2 = self.get_gray_img(self.curr_id)
        img1 = self.get_gray_img(self.prev_id)    #确保取到相邻两帧
        img0 = self.first_gray
        """
        diff = framediff(img0_d,img1)
        z = '000000'
        index = self.curr_id
        cv2.imwrite("D:/code/trackbypredict/results/diff-plot/" + z[0:-len(str(index))] + str(index)+ '.jpg', diff)
        """
        is_exist = np.ones(len(self.curr_box))

        #for i, (b0_xywh, b_xywh) in enumerate(zip(boxes0, boxes)):  #for every box
        #print(boxes,self.NeededFix)
        for i, b_xywh in enumerate(boxes):  #for every box
            #if self.cls[i] != -1:
            #    continue
            #b0_xyxy, b_xyxy = xywh2xyxy(b0_xywh), xywh2xyxy(b_xywh)
            #area0 = img0[int(b0_xyxy[1]*fh):int(b0_xyxy[3]*fh),int(b0_xyxy[0]*fw):int(b0_xyxy[2]*fw)]
            #area = img1[int(b_xyxy[1]*fh):int(b_xyxy[3]*fh),int(b_xyxy[0]*fw):int(b_xyxy[2]*fw)]

            #zncc = template_match_ZNCC(area0,area)
            #print(zncc)

            #1.假设预先知道所有iou，修正效果如何
            #test_iou = box_iou_np(np.expand_dims(xywh2xyxy(b_xywh),0), xywh2xyxyn(np.array(self.gt)[:,1:]))[0]
            #if max(test_iou) < 0.7: #and max(test_iou) > 0.25 or max(test_iou)==0:   
                 
            #if zncc < 0.70:
            #if True:
            #if  b_xywh[1]>=0.4 or (b_xywh[2] > 0.05 and b_xywh[3] > 0.1):   #data2
            #if  b_xywh[1]>=0.6:
            #if  b_xywh[1]>=0.55 or (b_xywh[2] > 0.05 and b_xywh[3] > 0.1):   #data3
            #if  b_xywh[1]>=0.85:   #data3
            #先检查框是否合法，不合法的改为合法的
            b_xywh = xyxy2xywh(cut_outside(xywh2xyxy(b_xywh)))
            if if_in(params['left_area'][0],(b_xywh[0],b_xywh[1])):   #data1
                self.NeededFix[i] = 1
                
            #if b_xywh[0] >= 0.65:
            #    self.NeededFix[i] = 1
            if self.NeededFix[i]:
            #if b_xywh[1] > 0.32:
                
                if not if_legal(b_xywh):
                    left = 1
                    fix_xywh = b_xywh
                else:
                    fix_xywh, left = fix_box(img0_d, img1, img2, b_xywh, boxes, boxes_p, self.curr_id)
                    #fix_xywh, left = fix_box(img0_d, img1, img2, b_xywh, boxes)     #pareto
                """
                fix_xywh,left = np.array(self.gt)[:,1:][np.argmax(test_iou)], 0
                if max(test_iou) == 0:
                    left = 1
                """
                print("@@@:",b_xywh,fix_xywh,left)
                if left:
                    is_exist[i] = 0
                boxes[i] = fix_xywh
                self.NeededFix[i] = 0

        self.curr_box = boxes[is_exist==1]
        self.cls = self.cls[is_exist==1]
        self.NeededFix = self.NeededFix[is_exist==1]
        
        self.isNewBox = self.isNewBox[is_exist==1]
        #self.pprev_box = self.pprev_box[is_exist==1]
        
        if len(self.pred_mask) and len(self.pred_output):
            self.pred_output = self.pred_output[is_exist[self.pred_mask]==1,:,:]
            self.pred_mask = self.pred_mask[is_exist==1]     
        
        t2 = time.time()
        return 0
    

def choose_tracklet(label,speed,feature,num,random=False):

    if random:
        cost = creat_matrix(label,feature,fw,fh)     #random choose
    else:
        cost, vars = get_cost(label,speed,feature)

    total_cost, count = np.zeros(len(label)), np.zeros(len(label))
    col_inds = []
    for j in range(num):
        cost = np.nan_to_num(cost, nan=200, posinf=200, neginf=200)
        row_ind, col_ind = linear_sum_assignment(cost) 
        
        col_inds.append(col_ind)
        for i in range(len(row_ind)):
            
            if cost[row_ind[i]][col_ind[i]] < 200:
                total_cost[i] += cost[row_ind[i]][col_ind[i]]
                count[i] += 1
                cost[:, col_ind[i]] = 201
            else:   #没有在该框内找到合适的点
                col_ind[i] = -1
    #print(label, total_cost/count,count)
    return row_ind,np.array(col_inds).T,vars

def get_cost(label,speed,feature):
    #given a window, return the index of best traj of every box
    #label from detect the 1st frame
    #feature from tracker [num_features, track_len, 2]
    t1 = time.time()
    matrix = creat_matrix(label,feature,fw,fh)

    move_thrd = 1
    track_len = feature.shape[1]
    vars = np.zeros(len(label))
    #print(label)
    for i in range(len(matrix)):    #for one bounding box

        #sp = speed[i]*np.array([fw,fh])

        index = np.array(np.where(matrix[i]==10)[0])
        #print(label[i], feature[index,:,:])
        tracklets = feature[index,:,:]

        dis = tracklets[:,track_len-1,:] - tracklets[:,0,:]    #求位移 
        avg_dis = np.mean(dis)
        dis_x, dis_y = dis[:,0], dis[:,1]
        dis = np.linalg.norm(dis, axis=1)   #位移量

        #dis_var偏大多半出错，这里没有归一化，受尺度影响，先不管
        mean_dis = np.mean(dis)
        norm_length = dis/mean_dis
        mean_dis_x = np.mean(dis_x)
        mean_dis_y = np.mean(dis_y)
        
        dis_var = (np.std((dis_x/dis*norm_length))+np.std((dis_y/dis*norm_length)))/2
        #print("!!!!!",dis_x,dis_y,(dis_x/dis*norm_length),(dis_y/dis*norm_length))

        vars[i] = dis_var

        if len(tracklets) == 1: #框内只有一个点
            matrix[i][index]=0
        if len(tracklets) > 1:
            #归一化     
            min_dis_x = min(dis_x)
            min_dis_y = min(dis_y)             
            dis_x_n = (dis_x-min_dis_x)/(max(dis_x)-min_dis_x)  
            dis_y_n = (dis_y-min_dis_y)/(max(dis_y)-min_dis_y)
            
            moved_id = np.where(dis>move_thrd)[0]

            #moved_box = any(len([x for x in matrix[:,index[0][mid]] if x != 200])==1 for mid in moved_id)
            #print(label[i],"moved",len(moved_id),"not moved",len(tracklets)-len(moved_id))
            if len(moved_id) >= len(tracklets)-len(moved_id):# or moved_box: #moved
                #得分：与平均移动的差
                mean_dis_x_n = np.mean(dis_x_n)
                mean_dis_y_n = np.mean(dis_y_n)
                
                score_x = np.abs(dis_x_n-mean_dis_x_n)  
                score_y = np.abs(dis_y_n-mean_dis_y_n)
                
                #arc = np.arctan(dis_y/dis_x)
                #arc_p = np.arctan(sp[1]/sp[0]) if sp[0]!=0 else 0
                #score_a = np.abs(arc - arc_p)

                score = score_x + score_y #+ score_a
                
                matrix[i][index[moved_id]] = score[moved_id]
                # vars[i] = np.var(score[moved_id])
                #print("!!",i,vars[i])
                #print(score[moved_id],dis_x,dis_y)
    #print("getcost",time.time()-t1)
    return matrix, vars
