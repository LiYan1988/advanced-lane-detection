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
        
    def find_lane(self, img, reset=False):
        img = self.undistort(img)
        img = self.warp(img)
        
        img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        img_hls = cv2.medianBlur(img_hls, 5)
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_lab = cv2.medianBlur(img_lab, 5)
        
        big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        
        greenery = (img_lab[:, :, 2].astype(np.uint8) > 130) &\
            cv2.inRange(img_hls, (0, 0, 50), (35, 190, 255))
            
        road_mask = np.logical_not(greenery).astype(np.uint8) & \
            (img_hls[...,1]<250)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, small_kernel)
        road_mask = cv2.dilate(road_mask, big_kernel)
        
        _, contours, _ = cv2.findContours(road_mask, cv2.RETR_LIST, 
                                                     cv2.CHAIN_APPROX_NONE)
        
        biggest_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area>biggest_area:
                biggest_area = area
                biggest_contour = contour
        road_mask = np.zeros_like(road_mask)
        cv2.fillPoly(road_mask, [biggest_contour],  1)
        
        self.roi_mask[:, :, 0] = (self.left_line.line_mask | 
                self.right_line.line_mask) & road_mask
        self.roi_mask[:, :, 1] = self.roi_mask[:, :, 0]
        self.roi_mask[:, :, 2] = self.roi_mask[:, :, 0]
            
        return self.roi_mask
    
img = cv2.imread('test_images/straight_lines2.jpg')
#img = cv2.imread('test_images/test1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
lane_finder = LaneFinder((1280, 720), (600, 500), camera_matrix,
                         distortion_coefficients, perspective_matrix,
                         perspective_matrix_inverse, meter_per_pixel_x,
                         meter_per_pixel_y)

img_out = lane_finder.find_lane(img)
plt.figure(figsize=(16, 9))
plt.imshow(img_out)