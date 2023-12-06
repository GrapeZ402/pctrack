import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=int, choices=[0,1,2,3], default=1)
parser.add_argument('--delay', type=int, choices=[0,1,2,3,4], default=1)
parser.add_argument('--method', type=str, choices=['ours', 'base'], default='ours')
parser.add_argument('--source', type=str, default='datasets/OnRamp/')
parser.add_argument('--label', type=str, default='datasets/labels/OnRamp_label/')
parser.add_argument('--save', type=bool, default=False)

args = parser.parse_args()
data_param = args.data
delay_param = args.delay
method = args.method
save = args.save
source = args.source
label_path = args.label


WINDOW_LIST = [25,30,35,40,45]
feature_quality_level = 0.2
downsample = 5
FPS = 30

if data_param==2:
    fw,fh = 1280, 720
else:
    fw,fh = 1920, 1080

data_n = 0

weight_path_l = ['weights/Intersect.pth','weights/OnRamp.pth','weights/Express.pth','weights/UrbMixed.pth']
result_path_l = 'runs/result'

WINDOW = WINDOW_LIST[delay_param]

weight_path = weight_path_l[data_param]
result_path = result_path_l + str(data_param+1) + '-' + method + '_' + str(WINDOW) + '/'


#device
if 0:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

#detector
imgsz = (1920,1080)
detect_model = 'apps/yolov5/weights/yolov5m.pt'

params = {}
with open('data/OnRamp.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split(': ')
        if 'np.array' in value:
            value = value.replace('np.array', '').replace('(', '').replace(')', '')
            value = np.array(eval(value))
        else:
            value = eval(value)
        params[key] = value

