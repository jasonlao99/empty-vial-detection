import os 
import shutil
import cv2 as cv
from datetime import datetime
import numpy as np
import sys

########################################################################################################################

# user set values 
outputFileName = 'output.csv'
emptyOrFilled = False # false if empty and true if filled
path = "C:/Users/foo/"

########################################################################################################################

# constants
img_counter = 0
correct = 0
total = 0
BottleID = 0
classes = ['empty', 'not empty']

# set training file location
net = cv.dnn.readNetFromONNX('EVD4-27.onnx')

# redirect output
original_stdout = sys.stdout

# create and open output file
f = open(path+outputFileName, 'w')
header = "Date \t \t Timestamp \t BottleID \t Prediction \t Prediction Level"
header_csv = "Date, Timestamp,BottleID,Prediction,Prediction Level\n"
f.write(header_csv)
print(header)

# create "wrong" folder to hold incorrectly identified images
names = os.listdir(path)
folder_name = ['wrong']
for name in folder_name:
    if not os.path.exists(path+name):
        os.makedirs(path+name)

# process all PNG files in directory (change to .jpg if necessary)
for file in names:
    if ".png" in file:
        # process image with model
        image = cv.imread(path+file)
        image = cv.resize(image, (256, 256), interpolation=cv.INTER_AREA)
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

        # test filled images = 1
        # test empty images = 0
        if biggest_pred_index == emptyOrFilled:
            correct +=1
        else:
            # copy incorrect image to “wrong” folder
            shutil.copy(path+file, path+'wrong/'+file)
        total +=1
        # write output to file and console
        f.write(csv)
        print(s)

        # increment bottle ID 
        BottleID += 1

# close output file
f.close()

# print classification performance of model to console
per = (correct/total) * 100
print ("Performance (%)")
print (str(per))