import os 
import shutil
import random

path = "C:/Users/yxion/Documents/6_VandyUndergrad/4_ProgramManagement/EVD/Software/empty-vial-detection/camera/Not_Empty/"
names = os.listdir(path)
folder_name = ['training', 'validation', 'testing']
for name in folder_name:
    if not os.path.exists(path+name):
        os.makedirs(path+name)
for file in names:
    if ".png" in file: 
        num = random.randint(0,9)
        if num < 7: 
            shutil.move(path+file, path+'training/'+file)
        elif num < 9: 
            shutil.move(path+file, path+'validation/'+file)
        else: 
            shutil.move(path+file, path+'testing/'+file)
