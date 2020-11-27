# -*- coding: utf-8 -*-
"""
@author: Simone Diolaiuti

Copyright (c) 2020 Simone Diolaiuti

Additional Copyright notice: 
    Most of these functions are derived from the ones 
    provided as part of the course "Self Driving Car Nanodegree"
    by Udacity
                
MIT License

Copyright (c) 2016-2019 Udacity, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


"""

import cv2
import numpy as np

def grayscale(img):
    """
    Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussianBlur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def keepRegionOnly(img, vertices):
    """    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    
    Args:
        vertices: a numpy array of integer points describing 
        a polygon representing the image area you want to keep
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def drawLines(img, lines, color=[255, 0, 0], thickness=2):
    """   
    Draws `lines` with `color` and `thickness` on the image img provided.    
    
    Args:
        lines: list of two point coordinates representing the end points of each line to draw 
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
def drawPoints(img, points, color=[255, 0, 0], thickness=0):
    for point in points:
        cv2.circle(img,(point[0],point[1]),color=color,radius=0,thickness=thickness)
        
def drawPolyline(img, points, color=[255, 0, 0], thickness=2):
    prev_p = points[0]
    for p in points[1:]:
        cv2.line(img,(prev_p[0],prev_p[1]),(p[0],p[1]),color,thickness)
        prev_p = p
    #cv2.polylines(img, [points], isClosed=False, color=color, thickness=thickness)

def computeHoughLines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Args:
        img: a gray image highligting the edges of the original image
        
    Returns:
        list of lines found using Hough transformation; 
        each entry of the list is 4 elements: [x1,y1,x2,y2] 
        representing the coordinates of the end points of the line
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def blendImg(img_a, img_b, α=0.8, β=1., γ=0.):
    """    
    The result image is computed as follows:
    img_a * α + img_b * β + γ
    """
    return cv2.addWeighted(img_a, α, img_b, β, γ)

