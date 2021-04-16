import numpy as np
import cv2 as cv
import os
# Uncomment and add directory if dll cannot be found
#os.add_dll_directory(r'C:\Users\jason\Documents\Vanderbilt\4th Year Courseload\EECE Senior Design\Project\empty-vial-detection\camera')
import ctypes as C
import tisgrabber as IC
from datetime import datetime
import sys

# constants
img_counter = 0
BottleID = 0
classes = ['empty', 'not empty']

print("press space to take image, escape to end program")

# Create the camera object.
Camera = IC.TIS_CAM()

# List availabe devices as unique names. This is a combination of camera name and serial number
# Devices = Camera.GetDevices()
# for i in range(len( Devices )):
#     print( str(i) + " : " + str(Devices[i]))

# Open a device with hard coded unique name:
Camera.open("DFK Z12G445 37020482")

# or show the IC Imaging Control device page:
# Camera.ShowDeviceSelectionDialog()

if Camera.IsDevValid() == 1:
    # Start the live video stream, but show no own live video window. We will use OpenCV for this.
    # set to 0 stop supress second window
    Camera.StartLive(0)

    # Set camera  properties

    # Exposure
    ExposureAuto = [1]
    Camera.GetPropertySwitch("Exposure", "Auto", ExposureAuto)
    print("Exposure auto : ", ExposureAuto[0])

    # In order to set a fixed exposure time, the Exposure Automatic must be disabled first.
    # Using the IC Imaging Control VCD Property Inspector, we know, the item is "Exposure", the
    # element is "Auto" and the interface is "Switch". Therefore we use for disabling:
    Camera.SetPropertySwitch("Exposure", "Auto", 0)
    # "0" is off, "1" is on.

    # Exposure Time
    ExposureTime = [0]
    Camera.GetPropertyAbsoluteValue("Exposure", "Value", ExposureTime)
    print("Exposure time abs: ", ExposureTime[0])

    # Set an absolute exposure time, given in fractions of seconds. 0.0303 is 1/30 second:
    # increase if dark 
    Camera.SetPropertyAbsoluteValue("Exposure", "Value", 0.0015)

    # Proceed with Gain, since we have gain automatic, disable first. Then set values.
    Gainauto = [0]
    Camera.GetPropertySwitch("Gain", "Auto", Gainauto)
    print("Gain auto : ", Gainauto[0])

    Camera.SetPropertySwitch("Gain", "Auto", 0)
    Camera.SetPropertyValue("Gain", "Value", 600)

    WhiteBalanceAuto = [0]
    Camera.SetPropertySwitch("WhiteBalance", "Auto", 0)
    Camera.GetPropertySwitch("WhiteBalance", "Auto", WhiteBalanceAuto)
    print("WB auto : ", WhiteBalanceAuto[0])

    # White Balance
    Camera.SetPropertyValue("WhiteBalance", "White Balance Red", 111)
    Camera.SetPropertyValue("WhiteBalance", "White Balance Green", 64)
    Camera.SetPropertyValue("WhiteBalance", "White Balance Blue", 112)

    # Zoom
    Zoomauto = [0]
    Camera.SetPropertySwitch("Zoom", "Auto", 0)
    Camera.GetPropertySwitch("Zoom", "Auto", WhiteBalanceAuto)
    Camera.SetPropertyValue("Zoom", "Value", 50)
    print("Zoom auto : ", WhiteBalanceAuto[0])

    # Focus 
    Focusauto = [0]
    Camera.SetPropertySwitch("Focus", "Auto", 0)
    Camera.GetPropertySwitch("Focus", "Auto", Focusauto)
    Camera.SetPropertyValue("Focus", "Value", 270)
    print("Focus auto : ", Focusauto[0])

    # set training file location
    net = cv.dnn.readNetFromONNX('EVDresnet152.onnx')

    # redirect output
    original_stdout = sys.stdout

    # create and open output file
    f = open('output.csv', 'w')
    header = "Date \t \t Timestamp \t BottleID \t Prediction \t Prediction Level"
    header_csv = "Date, Timestamp,BottleID,Prediction,Prediction Level\n"
    f.write(header_csv)
    print(header)

    while True:
        # Snap an image
        Camera.SnapImage()

        # Get the image
        frame = Camera.GetImage()

        # Apply some OpenCV function on this image
        frameS = cv.resize(frame, (640, 480))
        cv.imshow('Window', frameS)

        # exit when esc is pressed
        k = cv.waitKey(1)
        if k % 256 == 27:
            print("closing...")
            break
        # otherwise wait for space bar
        elif k % 256 == 32:

            # save image
            img_name = "img_{}.png".format(img_counter)
            cv.imwrite(img_name, frame)
            img_counter += 1

            # process image with model
            image = frame
            image = cv.resize(image, (256, 256), interpolation=cv.INTER_AREA)
            blob = cv.dnn.blobFromImage(image, 1.0/255, (256, 256), (0.485, 0.456, 0.406), swapRB=True, crop=False)
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

            # write output to file and console
            f.write(csv)
            print(s)
            
            # increment bottle ID 
            BottleID += 1

    # close everything and exit
    Camera.StopLive()
    cv.destroyWindow('Window')
    f.close()


else:
    print("No device selected")