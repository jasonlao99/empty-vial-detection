import os 
import shutil
import cv2 as cv
from datetime import datetime
import numpy as np
import sys

# constants
img_counter = 0
correct = 0
total = 0
BottleID = 0
classes = ['empty', 'not empty']
outputFileName = 'QMSI_EMPTY.csv'
path = "C:/Users/yxion/Documents/6_VandyUndergrad/4_ProgramManagement/EVD/Software/empty-vial-detection/camera/test/not empty/EVDnew/"

# set training file location
net = cv.dnn.readNetFromONNX('EVDnew.onnx')

# redirect output
original_stdout = sys.stdout

# create and open output file
f = open(path+outputFileName, 'w')
header = "Date \t \t Timestamp \t BottleID \t Prediction \t Prediction Level"
header_csv = "Date, Timestamp,BottleID,Prediction,Prediction Level\n"
f.write(header_csv)
print(header)

names = os.listdir(path)
folder_name = ['correct', 'wrong']
for name in folder_name:
    if not os.path.exists(path+name):
        os.makedirs(path+name)
for file in names:
    if ".jpg" in file: 
        # process image with model
        image = cv.imread(path+file)
        image = cv.resize(image, (256, 256), interpolation=cv.INTER_AREA)
        # for our images
        #blob = cv.dnn.blobFromImage(image, 1.0/255, (256, 256), (0.485, 0.456, 0.406), swapRB=True, crop=False)
        # for QMSI images
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        red = image[:,:,2].copy()
        blue = image[:,:,0].copy()
        image[:,:,0] = red
        image[:,:,2] = blue
        image = image/255.0
        image = (image - mean)/std
        image = np.float32(image)
        blob = cv.dnn.blobFromImage(image, 1.0, (256, 256), (0,0,0), swapRB=False, crop=False)
        net.setInput(blob)
        preds = net.forward()
        biggest_pred_index = np.array(preds)[0].argmax(0)

        # get date and time
        now = datetime.now()
        date = now.strftime("%m/%d/%Y")
        timestamp = now.strftime("%H:%M:%S")

        # format output
        s = "{} \t {} \t {} \t\t {} \t {:.6f}".format(date, timestamp, BottleID, classes[biggest_pred_index].ljust(11), np.array(preds)[0][biggest_pred_index])
        csv = "{},{},{},{},{:.6f}\n".format(date, timestamp, BottleID, classes[biggest_pred_index], np.array(preds)[0][biggest_pred_index])

        # test empty images
        if biggest_pred_index == 0:
            correct +=1
        else:
            shutil.move(path+file, path+'wrong/'+file)
        total +=1
        # write output to file and console
        f.write(csv)
        print(s)

        # increment bottle ID 
        BottleID += 1
f.close()
per = (correct/total) * 100
print ("Performance (%)")
print (str(per))