import numpy as np
import cv2 as cv
import os
# Uncomment and add directory if dll cannot be found
#os.add_dll_directory(r'C:\Users\jason\Documents\Vanderbilt\4th Year Courseload\EECE Senior Design\Project\empty-vial-detection\camera')
import ctypes as C
import tisgrabber as IC
from datetime import datetime
import sys

# Code taken from https://github.com/TheImagingSource/IC-Imaging-Control-Samples. Credit goes to TheImagingSource.

# Constants
img_counter = 0
# Placeholder value for bottleIDs.
BottleID = 0
classes = ['empty', 'not empty']

print("press space to take image, escape to end program")

# Create the camera object.
Camera = IC.TIS_CAM()

# Run the following segment to open the camera selector menu. If the device ID is known, the camera can be directly
# opened with its unique hard coded name.

# List available devices as unique names. This is a combination of camera name and serial number
# Devices = Camera.GetDevices()
# for i in range(len( Devices )):
#     print( str(i) + " : " + str(Devices[i]))
# Show the IC Imaging Control device page:
# Camera.ShowDeviceSelectionDialog()

# Open a device with hard coded unique name:
Camera.open("DFK Z12G445 37020482")

if Camera.IsDevValid() == 1:
    # Start the live video stream, but show no own live video window. We will use OpenCV for this.
    # Set to 0 stop suppress second window
    Camera.StartLive(0)

    # Set camera property values accordingly
    # Auto = 0 means manual settings/control, 1 means the camera will automatically determine values
    # Manual setup is recommended
    # Ideal properties can be found using IC Capture's manual adjustments, then values can be added to here.
    # Property names can be found using IC Imaging Control VCD Property Inspector
    exposure_auto = 0
    exposure_time = 0.005

    gain_auto = 0
    gain_value = 7.27

    white_balance_auto = 0
    white_balance_red = 107
    white_balance_green = 67
    white_balance_blue = 200

    zoom_auto = 0
    zoom_value = 65

    focus_auto = 0
    focus_value = 286

    # Exposure
    ExposureAuto = [1]
    Camera.SetPropertySwitch("Exposure", "Auto", exposure_auto)
    Camera.GetPropertySwitch("Exposure", "Auto", ExposureAuto)
    print("Exposure auto : ", exposure_auto)

    # Exposure Time
    ExposureTime = [0]
    Camera.SetPropertyAbsoluteValue("Exposure", "Value", exposure_time)
    Camera.GetPropertyAbsoluteValue("Exposure", "Value", ExposureTime)
    print("Exposure time abs: ", ExposureTime[0])

    # Gain
    GainAuto = [0]
    Camera.SetPropertySwitch("Gain", "Auto", gain_auto)
    Camera.GetPropertySwitch("Gain", "Auto", GainAuto)
    print("Gain auto : ", Gainauto[0])

    GainValue = [0]
    Camera.SetPropertyAbsoluteValue("Gain", "Value", gain_value)
    Camera.GetPropertyAbsoluteValue("Gain", "Value", GainValue)
    print("Gain : ", GainValue[0])

    # While Balance
    WhiteBalanceAuto = [0]
    Camera.SetPropertySwitch("WhiteBalance", "Auto", white_balance_auto)
    Camera.GetPropertySwitch("WhiteBalance", "Auto", WhiteBalanceAuto)
    print("WB auto : ", WhiteBalanceAuto[0])

    # White Balance
    Camera.SetPropertyValue("WhiteBalance", "White Balance Red", white_balance_red)
    Camera.SetPropertyValue("WhiteBalance", "White Balance Green", white_balance_green)
    Camera.SetPropertyValue("WhiteBalance", "White Balance Blue", white_balance_blue)

    WhiteBalanceRed = [0]
    WhiteBalanceBlue = [0]
    WhiteBalanceGreen = [0]
    Camera.GetPropertyValue("WhiteBalance", "White Balance Red", WhiteBalanceRed)
    Camera.GetPropertyValue("WhiteBalance", "White Balance Green", WhiteBalanceGreen)
    Camera.GetPropertyValue("WhiteBalance", "White Balance Blue", WhiteBalanceBlue)
    print("WB red : ", WhiteBalanceRed[0])
    print("WB green : ", WhiteBalanceBlue[0])
    print("WB blue : ", WhiteBalanceGreen[0])

    # Zoom
    ZoomAuto = [0]
    Camera.SetPropertySwitch("Zoom", "Auto", zoom_auto)
    Camera.GetPropertySwitch("Zoom", "Auto", ZoomAuto)
    print("Zoom auto : ", ZoomAuto[0])

    ZoomValue = [0]
    Camera.SetPropertyValue("Zoom", "Value", zoom_value)
    Camera.GetPropertyValue("Zoom", "Value", ZoomValue)
    print("Zoom value : ", ZoomValue[0])

    # Focus 
    FocusAuto = [0]
    Camera.SetPropertySwitch("Focus", "Auto", focus_auto)
    Camera.GetPropertySwitch("Focus", "Auto", FocusAuto)
    print("Focus auto : ", FocusAuto[0])

    FocusValue = [0]
    Camera.SetPropertyValue("Focus", "Value", focus_value)
    Camera.GetPropertyValue("Focus", "Value", FocusValue)
    print("Focus value : ", FocusValue[0])




    # Read training file from file location
    net = cv.dnn.readNetFromONNX('EVD.onnx')

    # Create and open output file
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

        # Resize frame to display better on smaller computer screens
        frameS = cv.resize(frame, (640, 480))
        cv.imshow('Window', frameS)

        # Exit when esc is pressed (the program will not stop without escape)
        k = cv.waitKey(1)
        if k % 256 == 27:
            print("closing...")
            break
        # Otherwise wait for space bar
        elif k % 256 == 32:

            # Save image
            img_name = "4_17_contrast_2_img_{}.png".format(img_counter)
            cv.imwrite(img_name, frame)
            img_counter += 1

            # Process image with model
            image = frame
            # Resize image
            image = cv.resize(image, (256, 256), interpolation=cv.INTER_AREA)

            # Load in mean, std, used in training
            # These values are from ImageNet, and are used as default values
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            # Swap red and blue as OpenCV is BGR not RGB
            red = image[:,:,2].copy()
            blue = image[:,:,0].copy()
            image[:,:,0] = red
            image[:,:,2] = blue
            # Normalize image values between 0 and 1
            image = image/255.0
            # Normalize image using mean and std
            image = (image - mean)/std
            # Convert value to float and create a blob
            image = np.float32(image)
            blob = cv.dnn.blobFromImage(image, 1.0, (256, 256), (0,0,0), swapRB=False, crop=False)

            # Read the image into the model and find the prediction
            net.setInput(blob)
            preds = net.forward()
            biggest_pred_index = np.array(preds)[0].argmax(0)

            # Get date and time
            now = datetime.now()
            date = now.strftime("%m/%d/%Y")
            timestamp = now.strftime("%H:%M:%S")

            # Format output
            s = "{} \t {} \t {} \t\t {} \t {:.6f}".format(date, timestamp, BottleID, classes[biggest_pred_index].ljust(11), np.array(preds)[0][biggest_pred_index])
            csv = "{},{},{},{},{:.6f}\n".format(date, timestamp, BottleID, classes[biggest_pred_index], np.array(preds)[0][biggest_pred_index])

            # Write output to file and console
            f.write(csv)
            print(s)
            
            # Increment bottle ID (replace bottleIDs with the actual bottleID every image)
            BottleID += 1

    # close everything and exit
    Camera.StopLive()
    cv.destroyWindow('Window')
    f.close()


else:
    print("No device selected")