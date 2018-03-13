# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:14:59 2018

@author: home
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import cv2
import os
import numpy as np
import math

with open('camera_parameters.p', 'rb') as pfile:
    camera_parameters = pickle.load(pfile)

camera_matrix = camera_parameters['camera_matrix']
distortion_coefficients = camera_parameters['distortion_coefficients']
    
with open('perspective_data.p', 'rb') as pfile:
    perspective_data = pickle.load(pfile)
    
perspective_matrix = perspective_data['perspective_matrix']
perspective_matrix_inverse = perspective_data['perspective_matrix_inverse']
meter_per_pixel_x, meter_per_pixel_y = perspective_data['meter_per_pixel']
src_pts = perspective_data['src_pts']
dst_pts = perspective_data['dst_pts']

class LaneFinder:
    def __init__(self, image_size, warped_size, camera_matrix, 
                 distortion_coefficients, perspective_matrix,
                 perspective_matrix_inverse,
                 meter_per_pixel_x, meter_per_pixel_y):
        self.camera_matrix = camera_matrix
        self.found = False
        self.distortion_coefficients = distortion_coefficients
        self.image_size = image_size
        self.warped_size = warped_size
        self.blank_mask_warped = np.zeros((warped_size[1], warped_size[0], 3), 
                                           dtype=np.uint8)
        self.roi_mask = np.ones((warped_size[1], warped_size[0], 3), 
                                 dtype=np.uint8)
        self.total_mask = np.zeros_like(self.roi_mask)
        self.warped_mask = np.zeros((self.warped_size[1], 
                                     self.warped_size[0]), dtype=np.uint8)
        self.perspective_matrix = perspective_matrix
        self.perspective_matrix_inverse = perspective_matrix_inverse
        self.count = 0
#        self.left_line = 
    
    def undistort(self, img):
        return cv2.undistort(img, self.camera_matrix, 
                             self.distortion_coefficients)
        
    def warp(self, img):
        return cv2.warpPerspective(img, self.perspective_matrix,
                                   self.warped_size, 
                                   flags=cv2.WARP_FILL_OUTLIERS+
                                   cv2.INTER_CUBIC)
        
    def unwarp(self, img):
        return cv2.warpPerspective(img, self.perspective_matrix_inverse,
                                   self.image_size, flags=
                                   cv2.WARP_FILL_OUTLIERS +
                                   cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)
        
    def find_lane(self, img, *kwarg, reset=False):
        img = self.undistort(img)
        img = self.warp(img)
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv = cv2.medianBlur(img_hsv, 5)
        img_v = img_hsv[...,2]
        
        img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        img_hls = cv2.medianBlur(img_hls, 5)
        img_s = img_hls[...,2]
        
        # lab is good for yellow line detection
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_lab = cv2.medianBlur(img_lab, 5)
        img_b = img_lab[...,2]
        
        kwarg[0].append(img_v)
        kwarg[1].append(img_s)
        kwarg[2].append(img_b)

#        n_h = 10
#        h_size = img.shape[0]//n_h
#        for h in range(0, img.shape[0], h_size):
#            h_max = min(img.shape[0], h + h_size)
#            img_v_mean = img_v[h:h_max,...].mean()
#            img_v_std = img_v[h:h_max,...].std()
#            img_s_mean = img_s[h:h_max,...].mean()
#            img_s_std = img_s[h:h_max,...].std()
#            img_b_mean = img_b[h:h_max,...].mean()
#            img_b_std = img_b[h:h_max,...].std()
#            kwarg[0].append(img_v_mean)
#            kwarg[1].append(img_v_std)
#            kwarg[2].append(img_s_mean)
#            kwarg[3].append(img_s_std)
#            kwarg[4].append(img_b_mean)
#            kwarg[5].append(img_b_std)
#            
#            img_v[h:h_max,...] = cv2.inRange(img_v[h:h_max,...], img_v_mean + 2 * img_v_std, img_v_mean + 2.5 * img_v_std)
#            img_s[h:h_max,...] = cv2.inRange(img_s[h:h_max,...], img_s_mean + 7 * img_s_std, img_s_mean + 8.5 * img_s_std)
#            img_b[h:h_max,...] = cv2.inRange(img_b[h:h_max,...], img_b_mean + 2 * img_b_std, img_b_mean + 14 * img_b_std)
        
        img_v = cv2.inRange(img_v, 170, 230)
        img_s = cv2.inRange(img_s, 200, 245)
        img_b = cv2.inRange(img_b, 170, 200)
        
        img_sv = cv2.bitwise_or(img_s, img_v)
        img_out = cv2.bitwise_or(img_b, img_sv)
#        img_out = img_sv
        
        kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 19))
        img_out = cv2.morphologyEx(img_out, cv2.MORPH_DILATE, kernel)
        
        
        img_out = np.dstack((np.zeros_like(img_out), img_out, np.zeros_like(img_out)))
        img_out = cv2.addWeighted(img, 1, img_out, 1, 0)
        
        return img_out
    
img = cv2.imread('test_images/straight_lines1.jpg')
#img = cv2.imread('test_images/test5.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
lane_finder = LaneFinder((1280, 720), (600, 500), camera_matrix,
                         distortion_coefficients, perspective_matrix,
                         perspective_matrix_inverse, meter_per_pixel_x,
                         meter_per_pixel_y)

#img_s, img_b, img_v = lane_finder.find_lane(img)

#plt.plot(img_s.mean(axis=0))
#plt.show()
#plt.plot(img_b.mean(axis=0))
#plt.show()
#plt.plot(img_v.mean(axis=0))
#plt.show()

#plt.plot(img_s[150])
#plt.show()
#plt.plot(img_b[150])
#plt.show()
#plt.plot(img_v[150])
#plt.show()

#plt.figure(figsize=(16, 9))
#plt.imshow(img_sv, 'gray')
#plt.show()
#plt.figure(figsize=(16, 9))
#plt.imshow(img_b, 'gray')
#plt.show()


video_files = ['challenge_video.mp4']
output_path = "output_videos"
for file in video_files:
    output = os.path.join(output_path,"lane_"+file)
    clip2 = VideoFileClip(file)#.subclip(0,6)
    img_v = []
#    img_v_std = []
    img_s = []
#    img_s_std = []
    img_b = []
#    img_b_std = []
    challenge_clip = clip2.fl_image(lambda x: lane_finder.find_lane(x, img_v, img_s, img_b))
#    challenge_clip = clip2.fl_image(lambda x: lane_finder.find_lane(x))
    challenge_clip.write_videofile(output, audio=False)
    break


#%%
with open('vsb.p', 'wb') as f:
    pickle.dump({'v': img_v, 'b': img_b, 's': img_s}, f,
                pickle.HIGHEST_PROTOCOL)
    

#img_v_mean = np.array(img_v_mean)
#img_v_std = np.array(img_v_std)
#img_s_mean = np.array(img_s_mean)
#img_s_std = np.array(img_s_std)
#img_b_mean = np.array(img_b_mean)
#img_b_std = np.array(img_b_std)

#%%

#plt.figure(figsize=(16, 9))
#plt.plot(img_v_mean)
#plt.plot(img_v_mean + 2 * img_v_std)
#plt.plot(img_v_mean + 3 * img_v_std)
#plt.show()

#plt.figure(figsize=(16, 9))
#plt.plot(img_s_mean)
#plt.plot(img_s_mean + 9 * img_s_std)
#plt.plot(img_s_mean + 7 * img_s_std)
#plt.show()

#plt.figure(figsize=(16, 9))
#plt.plot(img_b_mean)
#plt.plot(img_b_mean + 14 * img_b_std)
#plt.plot(img_b_mean + 2 * img_b_std)
#plt.show()
    
#%%
import pandas as pd
with open('vsb.p', 'rb') as f:
    data = pickle.load(f)
data_length = len(data['v'])
for n in range(data_length):
    dfv = pd.DataFrame(data['v'][n])
    dfv.to_csv('csv/img_v_{}.csv'.format(n), header=None, index_col=None)
    dfb = pd.DataFrame(data['b'][n])
    dfb.to_csv('csv/img_b_{}.csv'.format(n), header=None, index_col=None)
    dfs = pd.DataFrame(data['s'][n])
    dfs.to_csv('csv/img_s_{}.csv'.format(n), header=None, index_col=None)
    break