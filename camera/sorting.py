import os 
import shutil
import random

# user specified path for folder
path = "C:/Users/foo/"

# obtain file names for everything that that directory
names = os.listdir(path)

# create necessary folders if they do not exist yet
folder_name = ['training', 'validation', 'testing']
for name in folder_name:
    if not os.path.exists(path+name):
        os.makedirs(path+name)

# sort all PNG files randomly into the folders w/ appropriate distribution
for file in names:
    if ".png" in file: 
        num = random.randint(0,9)
        if num < 7: 
            shutil.move(path+file, path+'training/'+file)
        elif num < 9: 
            shutil.move(path+file, path+'validation/'+file)
        else: 
            shutil.move(path+file, path+'testing/'+file)
