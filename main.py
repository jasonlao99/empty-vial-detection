import cv2
import numpy as np

net =  cv2.dnn.readNetFromONNX('EVD.onnx')
image = cv2.imread('E0000009.jpg')
blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (256, 256),(0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
preds = net.forward()
biggest_pred_index = np.array(preds)[0].argmax()
if biggest_pred_index == 1:
    print ("Predicted class: not empty")
else:
    print ("Predicted class: empty")