import torch
from utils import *
from apps.yolov5.models.common import *
from apps.yolov5.models.experimental import *
from apps.yolov5.utils.general import *
from apps.yolov5.models.yolo import *
from apps.yolov5.utils.datasets import LoadImages

class Detector():
    def __init__(self, imgsz) -> None:
        self.imgsz = imgsz
        self.model = DetectMultiBackend(detect_model, device=device)
        imgsz = check_img_size(self.imgsz, s=self.model.stride)

        self.model.model.float()
        self.model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup

    def yolov5_detect(self, im0):
        formatBoxes = []

        #preprocess
        im = letterbox(im0)[0]  #padded resize
        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.4, 0.45, [0,2], False, max_det=1000)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Write results
                for *xyxy, conf, cls in reversed(det):# Write to file
                    #print(xyxy,conf,cls)
                    xywh = (xyxy2xywhn(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    
                    conf = conf.tolist()
                    cls = cls.tolist()

                    formatBoxes.append([cls, *xywh])

        return im0,formatBoxes

