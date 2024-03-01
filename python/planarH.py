import numpy as np
import cv2
# np.set_printoptions(suppress=True) #For debugging

def computeH(x1, x2):
	#Q3.6
	#Compute the homography between two sets of points

	A = []
	for i in range(len(x1)):
		x_1, y_1 = x1[i][0], x1[i][1]
		x_2, y_2 = x2[i][0], x2[i][1]
		
		A.append([-x_2, -y_2, -1, 0, 0, 0, x_2*x_1, y_2*x_1, x_1])
		A.append([0, 0, 0, -x_2, -y_2, -1, x_2*y_1, y_2*y_1, y_1])

	A = np.array(A)
	U, S, Vt = np.linalg.svd(A)
	H2to1 = Vt[-1,:].reshape(3, 3)

	return H2to1


def computeH_norm(x1, x2):
	#Q3.7
	def normalize(pts):
		#Compute the centroid of the points
		centroid = np.mean(pts, axis=0)

		#Shift the origin of the points to the centroid
		shifted_pts = pts - centroid
		
		#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
		max_dist = np.max(np.linalg.norm(shifted_pts, axis=1))
		scale = np.sqrt(2) / max_dist

		# Transformation
		T = np.array([
			[scale, 0, -scale * centroid[0]],
			[0, scale, -scale * centroid[1]],
			[0, 0, 1]
		])

		homogeneous_pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
		normalized_homogeneous_pts = (T @ homogeneous_pts.T).T[:, :2]

		return T, normalized_homogeneous_pts

	#Normalize
	T1, x1_normalized = normalize(x1)
	T2, x2_normalized = normalize(x2)

	#Compute homography
	H_normalized = computeH(x1_normalized, x2_normalized)

	#Denormalization
	H2to1 = np.linalg.inv(T1) @ H_normalized @ T2

	return H2to1

def computeH_ransac(x1, x2, epochs=200, threshold=2):
	#Q3.8
	#Compute the best fitting homography given a list of matching points
	max_inliers = 0
	bestH2to1 = None
	best_inliers = np.zeros(len(x1), dtype=bool)

	for i in range(epochs):
		# Randomly sample 4 pairs of points
		indices = np.random.choice(len(x2), 4, replace=False)
		sample_x1 = x1[indices]
		sample_x2 = x2[indices]

		# Compute a homography from these points
		H = computeH_norm(sample_x1, sample_x2)

		# Apply the homography to all points in x2
		x2_homogeneous = np.concatenate((x2, np.ones((len(x2), 1))), axis=1)
		x1_projected_homogeneous = (H @ x2_homogeneous.T).T

		# Convert from homogeneous coordinates to 2D
		x1_projected = x1_projected_homogeneous[:, :2] / x1_projected_homogeneous[:, [2]]

		# Compute the inliers where the distance is below the threshold
		distances = np.sqrt(np.sum((x1 - x1_projected)**2, axis=1))
		inliers = distances < threshold

		# Update the best homography if this one has more inliers
		num_inliers = np.sum(inliers)
		if num_inliers > max_inliers:
			max_inliers = num_inliers
			bestH2to1 = H
			best_inliers = inliers

	# Convert boolean inliers mask to binary vector
	inliers = best_inliers.astype(int)

	return bestH2to1, inliers


def compositeH(H2to1, template, img):

	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	H_inv = np.linalg.inv(H2to1)

	# Create mask of same size as template
	mask = np.ones(template.shape, template.dtype)

	# Warp mask by appropriate homography
	warped_mask = cv2.warpPerspective(mask, H_inv, (img.shape[1], img.shape[0]))

	# Warp template by appropriate homography
	warped_template = cv2.warpPerspective(template, H_inv, (img.shape[1], img.shape[0]))

	# Use mask to combine the warped template and the image
	composite_img = img.copy()
	composite_img[warped_mask == 1] = warped_template[warped_mask == 1]

	return composite_img
