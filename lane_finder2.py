# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 08:59:28 2018

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
        self.lane_line_distance = 3.6576 # in meter
        self.meter_per_pixel_x = meter_per_pixel_x
        self.meter_per_pixel_y = meter_per_pixel_y
    
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
        
    def find_lane(self, img, reset=False):
        img = self.undistort(img)
        img = self.warp(img)
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv = cv2.medianBlur(img_hsv, 5)
        img_v = img_hsv[...,2]
        img_v = cv2.inRange(img_v, 230, 255)
        
        img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        img_hls = cv2.medianBlur(img_hls, 5)
        img_s = cv2.inRange(img_hls[...,2], 170, 220)
        img_l = cv2.inRange(img_hls[...,1], 200, 230)
        img_sl = cv2.bitwise_or(img_s, img_l)
        
        # lab is good for yellow line detection
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_lab = cv2.medianBlur(img_lab, 5)
        img_b = img_lab[...,2]
        img_b = cv2.inRange(img_b, 170, 250)
        
        img_out = cv2.bitwise_and(img_sl, img_v)
        img_out = cv2.bitwise_or(img_b, img_out)
        
        kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 15))
        img_out = cv2.morphologyEx(img_out, cv2.MORPH_DILATE, kernel)
        img_out = cv2.morphologyEx(img_out, cv2.MORPH_CLOSE, kernel)
        
        img_out = np.dstack((img_out, np.zeros_like(img_out), np.zeros_like(img_out)))
#        img_out = cv2.addWeighted(img, 1, img_out, 1, 0)
        
        return img_out, img
    
    def track_left_lane(self, img):
        if len(img.shape) > 2:
            img = img.mean(axis=2)
        histogram = img.mean(axis=0)
        
        x_right = np.argmax(histogram[:histogram.shape[0]//2])
        width = 80
        height = 20
        stride = 5
        x_list = []
        y_list = []
        for y in range(img.shape[0]-1, height-1, -stride):
            y_max = y
            y_min = y - height
            y_mid = int((y_max+y_min)/2)
            x_min = max(x_right - width//2, 0)
            x_max = x_right + width//2
            tmp = img[y_min:y_max, x_min:x_max]
            if tmp.sum() > 0:
                x_right = x_min + np.argmax(tmp.mean(axis=0))
            x_list.append(x_right)
            y_list.append(y_mid)
            
        y = np.array(y_list)
        x = np.array(x_list)
        
#        plt.imshow(img)
#        plt.plot(x, y)
#        plt.show()
        
        return y, x
    
    def fit_lane(self, y, x):
        fit_coefficients = np.polyfit(y, x, 2)
        y_samples = np.arange(0, self.warped_size[1])
        x_samples = np.polyval(fit_coefficients, y_samples)
        
        return fit_coefficients, y_samples, x_samples
        
    def search_right_lane(self, img, fit_coefficients, y_in, x_in):
        if len(img.shape) > 2:
            img = img.mean(axis=2)
        gradient = 2 * fit_coefficients[0] * y_in + fit_coefficients[1]
        delta_y = self.lane_line_distance*gradient/(self.meter_per_pixel_y*
            np.sqrt(1+gradient**2))
        delta_x = delta_y*(self.meter_per_pixel_y/
            (self.meter_per_pixel_x*gradient))
        delta_y = np.round(delta_y).astype(np.int32)
        delta_x = np.round(delta_x).astype(np.int32)
        y_out = y_in - delta_y
        x_out = x_in + delta_x
        x_out = np.round(x_out).astype(np.int32).ravel()
        y_out = np.round(y_out).astype(np.int32).ravel()
        
        width = 80
        height = 20
        stride = 5
        x_list = []
        y_list = []
        for n in range(x_out.shape[0]):
            x = x_out[n]
            y = y_out[n]
            if (y<self.warped_size[1]) and (y - height > 0):
                y_max = y
                y_min = y - height
                x_min = x - width//2
                x_max = x + width//2
                tmp = img[y_min:y_max, x_min:x_max]
                if tmp.sum() == 0:
                    x_right = x
                else:
                    x_right = x_min + np.argmax(tmp.mean(axis=0))
                x_list.append(x_right)
                y_list.append(y)
            
        y_list = np.array(y_list)
        x_list = np.array(x_list)
        
        
        return y_list, x_list
    
#    def calculate_curvature(fit_coefficients, y):
#        
    
    def detect_pipline(self, img):
        img_out, img_warped = self.find_lane(img)
        y_left, x_left = self.track_left_lane(img_out)
        fit_coefficients_left, y_left, x_left = self.fit_lane(y_left, x_left)
        y_right, x_right = self.search_right_lane(img_out, fit_coefficients_left, y_left, x_left)
        fit_coefficients_right, y_right, x_right = self.fit_lane(y_right, x_right)
        left = np.dstack((x_left, y_left)).squeeze()
        right = np.dstack((x_right, y_right)).squeeze()
        pts = np.append(left, right[::-1], axis=0).astype(np.int32)
        img_blank = np.zeros_like(img_warped, dtype=np.uint8)
        cv2.polylines(img_blank, [pts], True, (0, 255, 0), 10)
        cv2.fillPoly(img_blank, [pts], (0, 0, 255))
        img_poly = cv2.addWeighted(img_warped, 1, img_blank, 0.8, 0)
        
        img_unwarped = cv2.warpPerspective(img_blank, self.perspective_matrix_inverse, self.image_size)
        img_unwarped = cv2.addWeighted(img, 1, img_unwarped, 0.8, 0)
        return img_unwarped
        
        
#%%
img = cv2.imread('test_images/straight_lines2.jpg')
img = cv2.imread('test_images/test5.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
lane_finder = LaneFinder((1280, 720), (600, 500), camera_matrix,
                         distortion_coefficients, perspective_matrix,
                         perspective_matrix_inverse, meter_per_pixel_x,
                         meter_per_pixel_y)
#
#img_out = lane_finder.find_lane(img)
#y, x = lane_finder.track_left_lane(img_out)
#fit_coefficients, y_samples, x_samples = lane_finder.fit_left_lane(y, x)
#y, x = lane_finder.search_right_lane(img_out, fit_coefficients, y_samples, x_samples)
#
#plt.imshow(img_out[...,0], 'gray')
#plt.plot(x_samples, y_samples, linewidth=3)
#plt.plot(x, y, linewidth=5)
##
#img_out = lane_finder.detect_pipline(img)
##
#plt.imshow(img_out)
#plt.show()
#plt.imshow(grad_x, 'gray')`
#plt.show()
#plt.imshow(grad_mag, 'gray')
#plt.show()

#plt.plot(img_out.mean(axis=0).ravel())
#plt.plot([0.07]*600)

#%%
video_files = ['project_video.mp4']
#video_files = ['challenge_video.mp4']
output_path = "output_videos"
for file in video_files:
    output = os.path.join(output_path,"lane_"+file)
    clip2 = VideoFileClip(file)#.subclip(0,1)
    challenge_clip = clip2.fl_image(lambda x: lane_finder.detect_pipline(x))
    challenge_clip.write_videofile(output, audio=False)
    break