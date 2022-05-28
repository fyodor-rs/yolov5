import torch
import win32api, win32con
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression
import numpy as np
import pandas as pd
from utils.torch_utils import select_device
# import mouse
device=select_device('0')
weights = 'best.pt'
conf_thres=0.4
iou_thres=0.05

def loadModelOne():
    model = torch.hub.load('', 'custom', path=weights,source='local', device=device, force_reload=True)
    return model

def loadModelTwo():
    model = DetectMultiBackend(weights, device=device)
    model.warmup()  
    return model

def detectImgOne(model,img):
    # 檢測所有對象，返回結果
    # Detecting all the objects
    results = model(img, size=320).pandas().xyxy[0]
    return results

def detectImgTwo(model,img,sctArea):
     # 截屏图片转换
    im= letterbox(img,sctArea,stride=model.stride)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    # 检测目标
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  
    im /= 255  
    if len(im.shape) == 3:
            im = im[None]  

    results = model(im, augment=False, visualize=False)
    results = non_max_suppression(results, conf_thres, iou_thres, agnostic=False)
    return results    

def handleData(result):
    data=[]
    index=[]
    for i in result:
        for j in i:
            xmin,ymin,xmax,ymax,con,n = j 
            m=(float(xmin),float(ymin),float(xmax),float(ymax),float(con),0,'person')
            data.append(m)
            index.append(len(data)-1)  
    results=pd.DataFrame(data=data, index=index, columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"])       
    return results

def getScreenShotXY(isShotRatio,screenShotRatio,screenShotSize):
    w = win32api.GetSystemMetrics(0)
    h = win32api.GetSystemMetrics(1)

    if isShotRatio :
       sctArea = {"top": int(h // 2 * (1 - screenShotRatio[1])),
                  "left": int(w  // 2 * (1 - screenShotRatio[0])),
                  "width": int(w * screenShotRatio[0]),
                  "height": int(h * screenShotRatio[1])}
    else :
        sctArea = {"top": (h - screenShotSize[1]) // 2,
                   "left": (w - screenShotSize[0]) // 2,
                   "width":  screenShotSize[0],
                   "height": screenShotSize[1]}
    return sctArea

def lock(targets,headshot_mode,cWidth,cHeight,aaMovementAmp):
    xMid = round((targets.iloc[0].xmax + targets.iloc[0].xmin) / 2)
    yMid = round((targets.iloc[0].ymax + targets.iloc[0].ymin) / 2)
    box_height = targets.iloc[0].ymax - targets.iloc[0].ymin
    if headshot_mode:
        headshot_offset = box_height * 0.38
    else:
        headshot_offset = box_height * 0.2

    mouseMove = [xMid - cWidth, (yMid - headshot_offset) - cHeight]
    # cv2.circle(npImg, (int(mouseMove[0] + xMid), int(mouseMove[1] + yMid - headshot_offset)), 3, (0, 0, 255))
    # Moving the mouse
    if win32api.GetKeyState(0x14):
        # mouse.mouse_xy(int(mouseMove[0] * aaMovementAmp),int(mouseMove[1] * aaMovementAmp))
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(mouseMove[0] * aaMovementAmp), int(mouseMove[1] * aaMovementAmp), 0, 0)

