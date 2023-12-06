import os
from model.lstm import *
from model.cnn_lstm import *
from tracker import *
from utils.utils import *
from utils.config import *

import time
import psutil



class Timer:
    def __init__(self):
        self.start = time.time()
        self.now = time.time()
        self.delet = 0

    def sleep(self):
        time.sleep(0.001)
        self.now = time.time()
        return 0
    
    def delet_time(self,t):
        self.delet += t
        return 0
    
    def get_time(self):
        return time.time()-self.start-self.delet
    
    def get_realtime_id(self):
        t = time.time() - self.start -self.delet
        id = np.floor(t*30)+WINDOW+1
        return id


# @profile
def run(label_all, save_result = save):
  
    fid = 1+WINDOW   
    f_count = 0     #一个窗口的计数

    tracker = Tracker(label = label_all[fid-1-WINDOW], start_id = fid, method = method)
    timer = Timer()
    idp_t, track_t, move_t, fix_t, newo_t = 0,0,0,0,0
    nw = 20
   
    while fid <= WINDOW*(nw+1):
        
        #print("FID: ",fid)
        if f_count == 0 :    #30(0)
            DP_finished = 0
            t1 = time.time()
            if fid!=1+WINDOW:

                tracker.reset(label_all[fid-1-WINDOW], fid)    #⑧
            t2 = time.time()
            timer.delet_time(t2-t1)

            tracker.detectFeatures()
            #IDP
            
            if method == "ours":
                tracker.predict_IDP()
            elif method == "base":
                tracker.glimpse_IDP()
            #tracker.no_IDP()
            
            
            

            tt = time.time() - t2
            idp_t += tt
            print("IDP: ",timer.get_realtime_id(),"\ttime: ",tt)


        else:   #1-29
            #"""
            id = timer.get_realtime_id()
            if id < fid:
                timer.sleep()
                continue

            if id > fid:
                tracker.jump_frame()
                print("jump:  ",fid)
                          
            
            #"""
            else:
                t1 = time.time()
                if not DP_finished:
                    print("finish")
                    DP_finished = 1
                    
                    tracker.detectFeatures()
                    tracker.update_after_dp()
                    
                    #加一次optital flow & move
                    
                    tracker.OpticalFlow()
                    
                    #tracker.curr_id -= 2
                    #tracker.prev_id -= 2
                    
                    tracker.move_box()  
                    if method == "ours":
                        #tracker.det_newobj(newobj_area)   
                        tracker.diffixer()
               
                tracker.OpticalFlow()                   #③
                
                        
                        
                t2 = time.time()
                #移动目标框
                tracker.move_box()                          #⑤
                t3 = time.time()
                t4 = time.time()
                #修正/新目标检测

                if method == "ours":
                    if f_count%5 == 0: 

                        tracker.det_newobj(params['newobj_area'])                            #⑥
              
                        t4 = time.time() 

                        tracker.diffixer()                      #⑦

                
                tt1, tt2, tt3, tt4 = t2-t1, t3-t2, t4-t3, time.time()-t4
                track_t += tt1
                move_t += tt2
                fix_t += tt3
                newo_t += tt4
                print("track: ",fid,"\ttime: ",tt1+tt2+tt3+tt4,"(",tt1,tt2,tt3,tt4,")")

        #------------save-------------------
        t1 = time.time()
        result = tracker.getBboxResult()
        if save_result:
            save_txt(result, fid)
        t2 = time.time()
        timer.delet_time(t2-t1)
        #-------------------------------------
         
        f_count = (f_count + 1) % WINDOW
          
        fid += 1

    total_time = idp_t+ track_t+ move_t+ fix_t+ newo_t
    print("Avg time: ",total_time/nw, idp_t/nw, track_t/nw, move_t/nw, fix_t/nw, newo_t/nw)
    return 0


if __name__ == '__main__':
    total_time = 0
    for _,_,filenames in os.walk(label_path):
        filenames.sort()
        label_all = []
        for i,filename in enumerate(filenames) :
            frame_id=int(filename[0:6])
            label = []       
            with open(label_path+filename, "r") as f:
                for line in f.readlines():    #each line
                    line = line.strip('\n') 
                    #line = [float(x) for x in line_s.split()] 
                    
                    label.append(list(float(i) for i in line.split()))
            label_all.append(label)

    
    run(label_all)

