import numpy as np
import cv2
from matchPics import matchPics
from scipy.ndimage import rotate
import matplotlib.pyplot as plt


#Q3.5
#Read the image and convert to grayscale, if necessary
img = cv2.imread('../data/cv_cover.jpg')
if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize the histogram
histogram = np.zeros(36) 

for i in range(36):
	#Rotate Image
	rotated = rotate(img, 10*i, reshape=False)
	
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(img, rotated)

	#Update histogram
	histogram[i] = len(matches)


#Display histogram — For a result of only 3 orientations like in the write-up, I replaced 36s with 3 and 360 with 30
					#(these 3 oritations were chosen to emphasize the effect)
degrees = np.arange(0, 360, 10)
plt.figure(figsize=(10, 5))
plt.bar(degrees, histogram, width=8)
plt.title('Number of Matches for Each Rotation Angle')
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Number of Matches')
plt.xticks(degrees)
plt.show()

