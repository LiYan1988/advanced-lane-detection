# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 12:01:50 2018

@author: home
"""

import cv2
import numpy as np
import glob
import pickle
from matplotlib import pyplot as plt

#%% 
corner_list = []
for fname in glob.glob('camera_cal/calibration*.jpg'):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ret, corner = cv2.findChessboardCorners(img, (9, 6))
    cv2.drawChessboardCorners(img, (9, 6), corner, ret)
    if ret:
        plt.imshow(img)
        plt.title(fname)
        plt.show()
        corner_list.append(corner)
        
corner_list = list(np.squeeze(np.array(corner_list)))

#%%
object_points = np.zeros((9 * 6, 3), dtype=np.float32)
object_points[...,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
object_points = list(np.tile(object_points, (len(corner_list), 1, 1)))

#%%
_, camera_matrix, distortion_coefficients, _, _ = \
    cv2.calibrateCamera(object_points, corner_list, (1280, 720), None, None)
    
with open('camera_parameters.p', 'wb') as pfile:
    pickle.dump({'camera_matrix': camera_matrix, 
                 'distortion_coefficients': distortion_coefficients},
                pfile, pickle.HIGHEST_PROTOCOL)
    
