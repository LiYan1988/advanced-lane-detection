# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:00:56 2018

@author: home
"""

import cv2
import numpy as np
import glob
import pickle
from matplotlib import pyplot as plt

#%%
with open('camera_parameters.p', 'rb') as pfile:
    data = pickle.load(pfile)
camera_matrix, distortion_coefficients = \
    data['camera_matrix'], data['distortion_coefficients']

#%%
images = ['test_images/straight_lines2.jpg']
img = cv2.imread(images[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.undistort(img, camera_matrix, distortion_coefficients)

#plt.figure(figsize=(16, 9))
#plt.imshow(img)

#%% Find vanishing point
img_canny = cv2.Canny(img, 50, 200)
mask_roi = np.zeros_like(img_canny, np.uint8)
mask_roi = cv2.fillPoly(mask_roi, [np.array([[0, 670], [0, 450], [1279, 450], [1279, 670]])], (255,))
img_canny_roi = cv2.bitwise_and(img_canny, mask_roi)

#plt.figure(figsize=(16, 9))
#plt.imshow(img_canny_roi, 'gray')

#%%
plt.figure(figsize=(16, 9))
edges = cv2.HoughLinesP(img_canny_roi, 1, np.pi/180, 20, np.array([]), 100, 10)
img_canny_plot = np.dstack((img_canny_roi, img_canny_roi, img_canny_roi))

for line in edges:
    for x1, y1, x2, y2 in line:
        cv2.line(img_canny_plot, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
#plt.figure(figsize=(16, 9))
#plt.imshow(img_canny_plot)
#plt.show()
#%%
p = []
n = []
for line in edges:
    for x1, y1, x2, y2 in line:
        p.append((x1, y1))
        n.append((y1-y2, x2-x1))
p = np.array(p, dtype=np.float32).T
n = np.array(n, dtype=np.float32).T

a = np.linalg.inv(np.matmul(n, n.T))
b = np.matmul(n, np.diagonal(np.matmul(n.T, p)))
vp = np.round(np.matmul(a, b)).astype(np.int16)

img_vp = img.copy()
cv2.circle(img_vp, tuple(vp), 10, (0, 255, 0), 2)

#plt.figure(figsize=(16, 9))
#plt.imshow(img_vp)

#%%

def calculate_line(vp, point, ycoord):
    tan = (point[0] - vp[0])/(point[1] - vp[1])
    x = tan * (ycoord - vp[1]) + vp[0]
    return (x, ycoord)

top = np.float32(vp[1] + 60)
bottom = np.float32(img.shape[0] - 20)
width = np.float32(500)
p1 = (vp[0] - width/2, top)
p2 = (vp[0] + width/2, top)
p3 = calculate_line(vp, p2, bottom)
p4 = calculate_line(vp, p1, bottom)

src_pts = np.array([p1, p2, p3, p4], dtype=np.float32)
dst_pts = np.array([[0, 0], [500, 0], [500, 600], [0, 600]], dtype=np.float32)
perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
perspective_matrix_inverse = cv2.getPerspectiveTransform(dst_pts, src_pts)
img_perspective_transformed = cv2.warpPerspective(img, perspective_matrix, (600, 500))

#plt.figure(figsize=(16, 9))
#plt.imshow(img_perspective_transformed)
#plt.show()

#%%
img_perspective_transformed_hls = cv2.cvtColor(img_perspective_transformed, 
                                               cv2.COLOR_RGB2HLS)
plt.figure(figsize=(16, 9))
img_pth = img_perspective_transformed_hls[...,1]>128
img_pth[:, :100] = 0
img_pth[:, -200:] = 0

plt.figure(figsize=(16, 9))
plt.imshow(img_pth, 'gray')
plt.show()

m = cv2.moments(img_pth[:, :300].astype(np.uint8))
x_left = int(np.round(m['m10']/m['m00']))
m = cv2.moments(img_pth[:, 300:].astype(np.uint8))
x_right = int(np.round(m['m10']/m['m00'] + 300))
img_blank = np.zeros_like(img_perspective_transformed)
cv2.line(img_blank, (x_left, 0), (x_left, 500), (255, 0, 0), 5)
cv2.line(img_blank, (x_right, 0), (x_right, 500), (0, 255, 0), 5)
img_blank = cv2.addWeighted(img_blank, 0.8, img_perspective_transformed, 1, 0)
plt.imshow(img_blank)

meter_per_pixel_x = 3.6576/(x_right - x_left)
Lh = np.linalg.inv(np.matmul(perspective_matrix, camera_matrix))
meter_per_pixel_y = meter_per_pixel_x / (np.linalg.norm(Lh[:,0]) / np.linalg.norm(Lh[:,1]))
print(1/meter_per_pixel_x, 1/meter_per_pixel_y)

perspective_data = {'perspective_matrix': perspective_matrix,
                    'perspective_matrix_inverse': perspective_matrix_inverse,
                    'meter_per_pixel': (meter_per_pixel_x, meter_per_pixel_y),
                    'src_pts': src_pts,
                    'dst_pts': dst_pts}
with open('perspective_data.p', 'wb') as f:
    pickle.dump(perspective_data, f)
