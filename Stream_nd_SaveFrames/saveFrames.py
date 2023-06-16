import cv2
import numpy as np
import time
from keras.models import load_model
import tensorflow as tf


down_points = (224, 224)
size = 224
model = load_model('D:/Tutorial 7/Tutorial 7/Logs/model_at_epoch_5.h5')

cap = cv2.VideoCapture(0)
arr = np.array([])

result = "none"
i = 0
j = 0

frameList = []


def getOpticalFlow(videoList):

    startOpt = time.time()

    gray_video = []
    for i in range(len(videoList)):

        img = cv2.cvtColor(videoList[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img,(size,size,)))

    flows = []
    for i in range(0,len(videoList)-1):    
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i+1], None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
         
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        
        flow[..., 0] = cv2.normalize(flow[..., 0], None,0,255,cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None,0,255,cv2.NORM_MINMAX)
       
        flows.append(flow)
        
    flows.append(np.zeros((size,size,2)))

    endOpt = time.time()
    timeTaken = endOpt - startOpt
    print(f"Time Taken For OpticalFlow : {timeTaken} secs")  


    return np.array(flows, dtype=np.float32)



def Video2Npy(frameList, resize=(size,size)):

    startNpy = time.time()

    frameF = []
    frames = frameList

    try:
        for i in range(len(frames)-1):
            # cv2.imshow("Live Video", frames[i])
            frame = cv2.resize(frames[i],resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (size, size,3))
            frameF.append(frame) 

    except Exception as e:
        print("Error: ", e)
    finally:
        frames = np.array(frameF)
        print('-----------------------------------------')

    flows = getOpticalFlow(frames)
    
    result = np.zeros((len(flows),size,size,5))
    
    result[...,:3] = frames
    result[...,3:] = flows

    endNpy = time.time()
    timeTaken = endNpy - startNpy
    print(f"Time Taken For Video2Npy : {timeTaken} secs")  

    return result


while(True):

    ret, frame = cap.read()

    i += 1
    
    key = cv2.waitKey(1)
    if key == ord("q"):
        break


    if ret:
        
        croppedFrame = frame[0:480, 100:580]
        croppedFrame = cv2.resize(croppedFrame, down_points, interpolation= cv2.INTER_LINEAR)
        frameList.append(croppedFrame)

        cv2.imshow("Live Video", croppedFrame)

        if i == 65:
    
            j +=1
            startOpt = time.time()
            
            print(len(frameList))
            result = Video2Npy(frameList)

            result = np.uint8(result)
            result = np.float32(result)
            # myFile = open('sample.txt', 'r+')
            # print("Done : ", result)
            # result=result.reshape(224,1120)
            # np.savetxt(myFile, result)
            # myFile.close()


            endOpt = time.time()
            e = endOpt - startOpt
            frameList = []
            print(f"Total Time : {e} secs")

            result = tf.stack([result],axis=0) 
            print(f"Result of iteration {j} is {np.shape(np.array(result))}")
            
            print("Model Prediction -------------------------")
            prediction = model.predict_step(result)
            print(f"Prediction is {prediction}")
            i = 0


cap.release()

cv2.destroyAllWindows()
