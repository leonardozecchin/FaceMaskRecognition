import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

imagesTrainWM_dir = './archive/FaceMaskDataset/Train/WithMask/'
dirs = os.listdir(imagesTrainWM_dir)


dim = (100,100)
images = []
for filename in os.listdir(imagesTrainWM_dir):
    img = cv2.imread(os.path.join(imagesTrainWM_dir,filename))
    if img is not None:
        img = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
        images.append(img)
M = len(images)
print(M)
[r,c,ch] = images[0].shape

print(images[0].shape)
cv2.imshow('sample image',images[0])
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image





'''img = cv2.imread('/media/leonardo/PopStorage/popOS/UNI/ML/FaceMaskRecognition/archive/Face Mask Dataset/Test/WithMask/53.png')

cv2.imshow('sample image',img)
 
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image'''