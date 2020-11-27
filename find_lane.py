# -*- coding: utf-8 -*-
"""
@author: Simone Diolaiuti
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from uda_utils import *
from geom_utils import *

from sklearn.cluster import KMeans

def findLane(img):
    """
    Args:
        img: a colored img as opened by matplotlib.image.read function

    Returns:
        the image overlayed with the identification of the lane 
    """
    ysize = img.shape[0]
    xsize =  img.shape[1]
    gscale_img = grayscale(img)
    blurred_img = gaussianBlur(gscale_img,5)
    canny_img = canny(blurred_img,low_threshold=10,high_threshold=180)
    
    # create a crop region to isolate the area of the image where the road is expected to be
    crop_region = np.array([[(0,ysize), 
                             (0, ysize*0.95),
                             (xsize*0.50, ysize*0.60), 
                             (xsize, ysize*0.95), 
                             (xsize,ysize)]],dtype=np.int32)
    
    cropped_img = keepRegionOnly(canny_img,crop_region)
    
    # find the lines through Hough transformation
    hlines = computeHoughLines(img=cropped_img,rho=1,theta=np.pi/180,threshold=5,min_line_len=10,max_line_gap=10) 
    drawLines(cropped_img,hlines)
    
    # get the equivalent list of Line2D instances
    elines = enrichLines(hlines)
    
    # remove segments that are too horizontal
    minslope = np.deg2rad(15)
    relevant_lines = [l for l in elines 
                      if (l.arc2 > minslope or l.arc2 < -minslope)] 
    
    
    # from set of Line2D to set of points
    if len(relevant_lines) > 0:
        line_points = np.array(relevant_lines[0].getSamplePoints())
        for l in relevant_lines[1:]:
            # choosing to pick a number of points that is proportional to the length
            # of the line to let the longer lines count more than the short ones
            n_points = l.length/10
            points = l.getSamplePoints(n_points)
            line_points = np.append(line_points,points,axis=0)
            
    # I'm applaying clustering here using two fictitious line to guide the classification
    # of the real points towards left and right lanes
    # The fictitious (anchor) points are then removed after the K-Means clustering 
    # has completed
    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    left_points = np.empty(shape=(0,2),dtype=np.float32)
    right_points = np.empty_like(left_points)
    
    if len(line_points) > 0:
        anchor_line_left = Line2D.fromAB(0,0,0,ysize-1).getSamplePoints(50)
        anchor_line_right = Line2D.fromAB(xsize-1,0,xsize-1,ysize-1).getSamplePoints(50)
        n_anchor_points = len(anchor_line_left)+len(anchor_line_right)
        line_points = np.append(line_points,anchor_line_left,axis=0)
        line_points = np.append(line_points,anchor_line_right,axis=0)


        # allowing K-Means start with the initial centroids at the opposite sides
        starting_centroids = np.array([[0,ysize/2],[xsize-1,ysize/2]])

        clustered = KMeans(2,starting_centroids).fit(line_points)


        for i,p in enumerate(line_points[:-n_anchor_points]):
            if clustered.labels_[i] == 0:
                left_points = np.append(left_points,[p],axis=0)
            else:
                right_points = np.append(right_points,[p],axis=0)

        drawPoints(line_img,left_points.astype(np.uint32),[255,0,0])
        drawPoints(line_img,right_points.astype(np.uint32),[0,0,255])
    
    # compute polinomial regression over left points and draw it (in red)
    rpl = computeRegressionPoints(left_points,(0,xsize/2))
    drawPolyline(line_img,rpl.astype(np.uint32),[255,0,0],4)
    
    # compute polinomial regression over right points and draw it (in blue)
    rpl = computeRegressionPoints(right_points,(xsize/2,xsize))
    drawPolyline(line_img,rpl.astype(np.uint32),[0,0,255],4)
    
    #overlap lane lines over original image
    blended_img = np.zeros_like(line_img)
    blended_img = blendImg(img,line_img,0.8,0.5)
    
    final_img = blended_img
    
    return final_img






