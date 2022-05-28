import win32api, win32con
import dealway
import time
import mss
import cv2
import numpy as np
import gc

def apex():
    # 加载方式 
    hWay=2

    # 截屏比例
    screenShotRatio =(0.3,0.5)

    # 截屏大小
    screenShotSize =(320,320) 

    # 使用截屏比例
    isShotRatio =False
    
    # 自瞄區域
    aaDetectionBox = 640
    
    # 鼠標移動速度,建議為1,比較鷄肋的參數
    aaMovementAmp = 1
    
    # 人體特徵匹配度,用於過濾
    confidence = 0.8
    
    # 退出按鍵
    aaQuitKey = "P"
     
    # 是否苗頭
    headshot_mode = False
    
    # 是否顯示幀數
    cpsDisplay = True
    
    # 是否查看調試框
    visuals = True

    # 獲取截屏參數
    sctArea= dealway.getScreenShotXY(isShotRatio,screenShotRatio,screenShotSize)

    # 计算中心自动瞄准框，必須為檢測框 的1/2
    cWidth = sctArea['width'] / 2
    cHeight = sctArea['height'] / 2

    # 强制垃圾收集,計時器
    count = 0
    sTime = time.time()

    # 畫框顔色
    COLORS = np.random.uniform(0, 255, size=(1500, 3))
    
    # 加載模型
    if hWay==1:
       model = dealway.loadModelOne()
    else:   
       model = dealway.loadModelTwo()

    # 加載截圖引擎
    cap = mss.mss()

    # 截圖檢測,按下指定按鍵結束進程
    while win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0:

        # 獲取截圖
        # img = cv2.cvtColor(np.array(cap.grab(sctArea)), cv2.COLOR_BGRA2BGR)
        img = np.delete(np.array(cap.grab(sctArea)), 3, axis=2)

        # 開始檢測
        if hWay==1:
            results=dealway.detectImgOne(model,img)
        else:   
            result=dealway.detectImgTwo(model,img,(sctArea['width'],sctArea['height']))
            results=dealway.handleData(result)
    
        # 過濾除了人之外的東西,confidence為相似，匹配度
        filteredResults = results[(results['class']==0) & (results['confidence']>confidence)]
        # print(filteredResults)

        # 返回真/假数组，具体取决于它是否在中央自动瞄准框中
        cResults = ((filteredResults["xmin"] > cWidth - aaDetectionBox) & (filteredResults["xmax"] < cWidth + aaDetectionBox)) & ((filteredResults["ymin"] > cHeight - aaDetectionBox) & (filteredResults["ymax"] < cHeight + aaDetectionBox))

        targets = filteredResults[cResults]
        # 鼠标移动
        if len(targets) > 0:
          dealway.lock(targets,headshot_mode,cWidth,cHeight,aaMovementAmp)
        # 為每一個檢測目標畫框
        if visuals:
            for i in range(0, len(results)):
                (startX, startY, endX, endY) = int(results["xmin"][i]), int(results["ymin"][i]), int(results["xmax"][i]), int(results["ymax"][i])
                confidence = results["confidence"][i]
                idx = int(results["class"][i])
                # 畫框和標簽
                # draw the bounding box and label on the frame
                label = "{}: {:.2f}%".format(results["name"][i], confidence * 100)
                cv2.rectangle(img, (startX, startY), (endX, endY),COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(img, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # 每秒强制清理垃圾,獲取幀數
        count += 1
        if (time.time() - sTime) > 1:
            if cpsDisplay:
                print("CPS: {}".format(count))
            count = 0
            sTime = time.time()
            gc.collect(generation=0)

        # 是否開啓檢測窗口
        # See visually what the Aimbot sees
        if visuals:
            cv2.imshow('Live Feed', img)
            if (cv2.waitKey(1) & 0xFF) == ord('p'):
                exit()

if __name__ == "__main__":
    apex()