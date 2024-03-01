import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from helper import plotMatches

def matchPics(I1, I2):

	#Convert Images to GrayScale
	if I1.ndim == 3:
		I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	if I2.ndim == 3:
		I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	
	#Detect Features in Both Images
	locs1 = corner_detection(I1, 0.15)
	locs2 = corner_detection(I2, 0.15)
	
	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(I1, locs1)
	desc2, locs2 = computeBrief(I2, locs2)

	#Match features using the descriptors
	matches = briefMatch(desc1, desc2, 0.7)

	return matches, locs1, locs2

# -------- TESTING --------
# import matplotlib.pyplot as plt

# I1 = plt.imread('../data/cv_cover.jpg')
# I2 = plt.imread('../data/cv_desk.png')

# matches, locs1, locs2 = matchPics(I1, I2)

# # Visualize the matches
# plotMatches(I1, I2, matches, locs1, locs2)

# plt.show()