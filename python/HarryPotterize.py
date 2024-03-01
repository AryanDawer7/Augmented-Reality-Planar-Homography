import numpy as np
import cv2
import skimage.io
import skimage.color

# Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

def HarryPotterize():

    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')

    matches, locs1, locs2 = matchPics(cv_cover, cv_desk)

    x1, x2 = np.zeros(locs1.shape), np.zeros(locs2.shape)
    x1[:, 0], x1[:, 1] = locs1[:, 1], locs1[:, 0]
    x2[:, 0], x2[:, 1] = locs2[:, 1], locs2[:, 0]

    # Scaling addressing the question in 3.9.4
    x1[:,0] = x1[:,0] * hp_cover.shape[1]/cv_cover.shape[1]
    x1[:,1] = x1[:,1] * hp_cover.shape[0]/cv_cover.shape[0]

    locs1, locs2 = x1[matches[:, 0]], x2[matches[:, 1]]
    
    H, inliers = computeH_ransac(locs1, locs2)

    composite_img = compositeH(H, hp_cover, cv_desk)

    cv2.imwrite('HarryPotterized.jpg', composite_img)
    cv2.imshow('HarryPotterized', composite_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the HarryPotterize script
if __name__ == '__main__':
    HarryPotterize()
