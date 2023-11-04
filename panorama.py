import cv2
import numpy as np

# Read images
I1 = cv2.imread('images/campus_01.jpg')
I2 = cv2.imread('images/campus_02.jpg')

# Resize imaged to half of their original size
I1 = cv2.resize(I1, None, fx=0.25, fy=0.25)
I2 = cv2.resize(I2, None, fx=0.25, fy=0.25)


# Get the original height and width of the image
I_height, I_width = I1.shape[:2]

# Convert images to grayscale
gI1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
gI2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

# Visualize image
#cv2.imshow('Color Image', cv2.resize(gI1, (I_width//5, I_height//5)))
#cv2.waitKey(0)

# Create ORB detector and descriptor
orb = cv2.ORB_create()

# Detect and compute descriptors for the first image
P1, F1 = orb.detectAndCompute(gI1, None)

# Detect and compute descriptors for the second image
P2, F2 = orb.detectAndCompute(gI2, None)

# Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(F2, F1)
matches = sorted(matches, key=lambda x: x.distance)

# Get the matching points
matchedPoints2 = np.float32([P2[match.queryIdx].pt for match in matches])
matchedPoints1 = np.float32([P1[match.trainIdx].pt for match in matches])

# Estimate the transformation
M, _ = cv2.findHomography(matchedPoints2, matchedPoints1, cv2.RANSAC, 5.0)

# Warp the second image onto the first image to create a panorama
h1, w1 = gI1.shape
h2, w2 = gI2.shape

corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

corners2_transformed = cv2.perspectiveTransform(corners2, M)
corners = np.concatenate((corners1, corners2_transformed), axis=0)

x_min, y_min = np.int32(corners.min(axis=0).ravel())
x_max, y_max = np.int32(corners.max(axis=0).ravel())

transformed_offset = (-x_min, -y_min)
transformed_image = cv2.warpPerspective(I2, M, (x_max - x_min, y_max - y_min))
transformed_image[transformed_offset[1]:h1 + transformed_offset[1], transformed_offset[0]:w1 + transformed_offset[0]] = I1

# Display the resulting panorama
cv2.imshow('Panorama', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()