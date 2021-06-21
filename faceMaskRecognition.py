import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/media/leonardo/PopStorage/popOS/UNI/ML/FaceMaskRecognition/archive/Face Mask Dataset/Test/WithMask/53.png')

cv2.imshow('sample image',img)
 
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image