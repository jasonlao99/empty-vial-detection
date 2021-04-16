import numpy as np
import cv2 as cv
import os
#os.add_dll_directory(r'C:\Users\jason\Documents\Vanderbilt\4th Year Courseload\EECE Senior Design\Project\empty-vial-detection\camera')
#os.add_dll_directory(r'C:\Users\yxion\Documents\6_VandyUndergrad\4_ProgramManagement\EVD\Software\empty-vial-detection\camera')
import ctypes as C
import tisgrabber as IC
from datetime import datetime
import sys

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
    # cv2.namedWindow('Window', cv2.cv.CV_WINDOW_NORMAL)

    # Set a video format
    # Camera.SetVideoFormat("RGB32 (640x480)")

    # Set a frame rate of 30 frames per second
    # Camera.SetFrameRate( 30.0 )

    # Start the live video stream, but show no own live video window. We will use OpenCV for this.
    # set to 0 stop supress second window
    Camera.StartLive(0)

    # Set some properties
    # Exposure time

    ExposureAuto = [1]

    Camera.GetPropertySwitch("Exposure", "Auto", ExposureAuto)
    print("Exposure auto : ", ExposureAuto[0])

    # In order to set a fixed exposure time, the Exposure Automatic must be disabled first.
    # Using the IC Imaging Control VCD Property Inspector, we know, the item is "Exposure", the
    # element is "Auto" and the interface is "Switch". Therefore we use for disabling:
    Camera.SetPropertySwitch("Exposure", "Auto", 0)
    # "0" is off, "1" is on.

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
    # Same goes with white balance. We make a complete red image:
    Camera.SetPropertySwitch("WhiteBalance", "Auto", 0)
    Camera.GetPropertySwitch("WhiteBalance", "Auto", WhiteBalanceAuto)
    print("WB auto : ", WhiteBalanceAuto[0])

    # Camera.SetPropertySwitch("WhiteBalance", "Auto", 0)
    # Camera.GetPropertySwitch("WhiteBalance", "Auto", WhiteBalanceAuto)
    # print("WB auto : ", WhiteBalanceAuto[0])

    Camera.SetPropertyValue("WhiteBalance", "White Balance Red", 111)
    Camera.SetPropertyValue("WhiteBalance", "White Balance Green", 64)
    Camera.SetPropertyValue("WhiteBalance", "White Balance Blue", 112)


    Zoomauto = [0]
    # Same goes with white balance. We make a complete red image:
    Camera.SetPropertySwitch("Zoom", "Auto", 0)
    Camera.GetPropertySwitch("Zoom", "Auto", WhiteBalanceAuto)
    print("Zoom auto : ", WhiteBalanceAuto[0])

    Camera.SetPropertyValue("Zoom", "Value", 50)


    Focusauto = [0]
    # Same goes with white balance. We make a complete red image:
    Camera.SetPropertySwitch("Focus", "Auto", 0)
    Camera.GetPropertySwitch("Focus", "Auto", Focusauto)
    print("Focus auto : ", Focusauto[0])

    Camera.SetPropertyValue("Focus", "Value", 270)

    img_counter = 0

    correct = 0
    classes = ['empty', 'not empty']

    net = cv.dnn.readNetFromONNX('EVDresnet152.onnx')

    original_stdout = sys.stdout
    BottleID = 0

    f = open('output.txt', 'w')
    header = "Date \t \t Timestamp \t BottleID \t Prediction \t Prediction Level"
    f.write(header)
    f.write("\n")
    print(header)

    while True:
        # Snap an image
        Camera.SnapImage()
        # Get the image
        frame = Camera.GetImage()
        # Apply some OpenCV function on this image
        frameS = cv.resize(frame, (640, 480))
        cv.imshow('Window', frameS)

        k = cv.waitKey(1)
        if k % 256 == 27:
            print("closing...")
            break
        elif k % 256 == 32:
            img_name = "img_{}.png".format(img_counter)
            cv.imwrite(img_name, frame)
            # print("Picture {} saved.".format(img_name))
            img_counter += 1

            image = frame
            image = cv.resize(image, (256, 256), interpolation=cv.INTER_AREA)
            blob = cv.dnn.blobFromImage(image, 1.0/255, (256, 256), (0.485, 0.456, 0.406), swapRB=True, crop=False)
            net.setInput(blob)
            preds = net.forward()
            biggest_pred_index = np.array(preds)[0].argmax(0)
            # threshold = 2.5
            # idx = np.argsort(preds[0])[::-1][0]
            # scuff_conf = (preds[0][idx]/threshold) * 100
            # text = "Label: {}, {}".format(classes[idx], scuff_conf)
            now = datetime.now()
            date = now.strftime("%m/%d/%Y")
            timestamp = now.strftime("%H: %M: %S")
            s = "{} \t {} \t {} \t\t {} \t\t {:.6f}".format(date, timestamp, BottleID, classes[biggest_pred_index], np.array(preds)[0][biggest_pred_index])
            f.write(s)
            f.write("\n")

            BottleID += 1

            print(s)
            # print(text)
            #if biggest_pred_index == 1:
            #    print("Predicted class: not empty")
            #else:
            #    print("Predicted class: empty")


    Camera.StopLive()
    cv.destroyWindow('Window')
    f.close()


else:
    print("No device selected")