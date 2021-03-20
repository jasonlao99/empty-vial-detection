import numpy as np
import cv2 as cv
import ctypes as C
import tisgrabber as IC

print("press space to take image, escape to end program")

# Create the camera object.
Camera = IC.TIS_CAM()

# List availabe devices as uniqe names. This is a combination of camera name and serial number
Devices = Camera.GetDevices()
for i in range(len( Devices )):
    print( str(i) + " : " + str(Devices[i]))

# Open a device with hard coded unique name:
#Camera.open("DFK 33UX290 18810256")
# or show the IC Imaging Control device page:

Camera.ShowDeviceSelectionDialog()

if Camera.IsDevValid() == 1:
    # cv2.namedWindow('Window', cv2.cv.CV_WINDOW_NORMAL)
    print('Press ctrl-c to stop')

    # Set a video format
    # Camera.SetVideoFormat("RGB32 (640x480)")

    # Set a frame rate of 30 frames per second
    # Camera.SetFrameRate( 30.0 )

    # Start the live video stream, but show no own live video window. We will use OpenCV for this.
    Camera.StartLive(1)

    # Set some properties
    # Exposure time

    ExposureAuto = [1]

    Camera.GetPropertySwitch("Exposure", "Auto", ExposureAuto)
    print("Exposure auto : ", ExposureAuto[0])

    # In order to set a fixed exposure time, the Exposure Automatic must be disabled first.
    # Using the IC Imaging Control VCD Property Inspector, we know, the item is "Exposure", the
    # element is "Auto" and the interface is "Switch". Therefore we use for disabling:
    Camera.SetPropertySwitch("Exposure", "Auto", 1)
    # "0" is off, "1" is on.

    # ExposureTime = [0]
    # Camera.GetPropertyAbsoluteValue("Exposure", "Value", ExposureTime)
    # print("Exposure time abs: ", ExposureTime[0])
    #
    # # Set an absolute exposure time, given in fractions of seconds. 0.0303 is 1/30 second:
    # Camera.SetPropertyAbsoluteValue("Exposure", "Value", 0.0303)

    # Proceed with Gain, since we have gain automatic, disable first. Then set values.
    Gainauto = [0]
    Camera.GetPropertySwitch("Gain", "Auto", Gainauto)
    print("Gain auto : ", Gainauto[0])

    Camera.SetPropertySwitch("Gain", "Auto", 1)
    # Camera.SetPropertyValue("Gain", "Value", 10)

    WhiteBalanceAuto = [0]
    # Same goes with white balance. We make a complete red image:
    Camera.SetPropertySwitch("WhiteBalance", "Auto", 1)
    Camera.GetPropertySwitch("WhiteBalance", "Auto", WhiteBalanceAuto)
    print("WB auto : ", WhiteBalanceAuto[0])

    # Camera.SetPropertySwitch("WhiteBalance", "Auto", 0)
    # Camera.GetPropertySwitch("WhiteBalance", "Auto", WhiteBalanceAuto)
    # print("WB auto : ", WhiteBalanceAuto[0])

    # Camera.SetPropertyValue("WhiteBalance", "White Balance Red", 64)
    # Camera.SetPropertyValue("WhiteBalance", "White Balance Green", 64)
    # Camera.SetPropertyValue("WhiteBalance", "White Balance Blue", 64)

    img_counter = 0

    while True:
        # Snap an image
        Camera.SnapImage()
        # Get the image
        frame = Camera.GetImage()
        # Apply some OpenCV function on this image
        cv.imshow('Window', frame)

        k = cv.waitKey(1)
        if k % 256 == 27:
            print("closing...")
            break
        elif k % 256 == 32:
            img_name = "img_{}.png".format(img_counter)
            cv.imwrite(img_name, frame)
            print("Picture {} saved.".format(img_name))
            img_counter += 1

    Camera.StopLive()
    cv.destroyWindow('Window')


else:
    print("No device selected")