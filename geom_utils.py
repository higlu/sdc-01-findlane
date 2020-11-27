# -*- coding: utf-8 -*-
"""
Utility functions for geometric calculations

@author: Simone Diolaiuti
"""

import numpy as np
import math

def restrictToTwoQuadrants(alpha):
    """
    the same slope of a line can be represented by multiple angles.
    This function transform the provided angle representing a slope into 
    a the equivalent value within the range [+pi/2, -pi/2]
    
    Args:
        alpha: the angle in radiants representing the slope of a line
        
    Returns:
        equivalent slope within [+pi/2, -pi/2] range
    """
    while alpha > np.pi:
        alpha -= np.pi
    while alpha < -np.pi:
        alpha += np.pi
    while alpha > np.pi/2:
        alpha -= np.pi
    while alpha < -np.pi/2:
        alpha += np.pi
    return alpha

def pointDistance(p1,p2):
    """ computes the euclidian distance between two points """
    return math.sqrt(pow(p1[0]-p2[0],2)+pow(p1[1]-p2[1],2))

class Line2D:
    """
    A class to conveniently handle a line (segment) in 2D that can be built throug 
    various characterizations
    
    Attributes:
        pA:     point representing one end of the line (the one closest ot the 
                bottom of the image)
        pB:     the other end of the line
        length: the length of the line
        arc2:   the slope of the line in radiant [-pi/2,pi/2]
    """
    
    def __init__(self,pA,pB,length,arc4):
        """
        Args:
            pA: starting point of the line (closest to bottom side)
            pB: end point of the line (closest to upper side)
            center: center point of the line
            length: total length of the line
            arc4: slope in radiants (it could sit on any of the 4th quadrants)
        """
        self.pA = pA
        self.pB = pB
        self.length = length
        self.arc2 = restrictToTwoQuadrants(arc4)
        
        
    @staticmethod
    def getAB(x1,y1,x2,y2):
        """
        Returns:
            pA,pB: the two final points of the line, where pA is the closest 
                   to the bottom side of the image
        """
        
        p1 = [x1,y1]
        p2 = [x2,y2]
        if y1 > y2:
            pA = p1
            pB = p2
        else:
            pA = p2
            pB = p1
            
        return pA,pB
    
        
    @classmethod
    def fromAB(cls,x1,y1,x2,y2):
        """
        it constructs a Line2D instance through the coordinates of the 
        two end points of the line
        """
        pA,pB = cls.getAB(x1,y1,x2,y2)
        
        # now pB is the point closest to the horizon
        # and pA closest to the observer
        # assuming a landscape persepective
        
        # compute the center of mass
        cx = pB[0] + (pA[0]-pB[0])/2.
        cy = pB[1] + (pA[1]-pB[1])/2.
        cm = [cx,cy]
        
        # compute length
        length = pointDistance(pA,pB)
        
        # compute slope in radiant
        slope = np.arctan2(pA[1]-pB[1],pA[0]-pB[0])
        
        return cls(pA,pB,length,slope)
        
        
    @classmethod
    def fromCenter(cls,center,length,slope_rad):
        """
        it constructs a Line2D instance using the following info
        
        Args:
            center:    middle point of the line 
            length:    length of the line
            slope_rad: slope of the line from horizontal (radiants)
            
        """
        dx = length/2.*np.cos(slope_rad)
        dy = length/2.*np.sin(slope_rad)
        
        p1 = [center[0]+dx,center[1]+dy]
        p2 = [center[0]-dx,center[1]-dy]
        
        pA,pB = cls.getAB(p1[0],p1[1],p2[0],p2[1])
        
        return cls(pA,pB,length,slope_rad)    
        
    @staticmethod
    def getPointBetween(p1,p2,factor):
        """
        Computes the coordinates of a point belonging to the line connecting
        p1 and p2 and at a certain distant factor from p1

        Args:
            p1:     one of the end of the line
            p2:     the other end of the line
            factor: identify how far from p1 you want the point [0,1].
                    If factor is 0.3 the returned point will sit
                    30% distant from p1 and 70% distant from p2

        Returns:
            pr:     the point at the given distant factor

        """
        dx = p2[0]-p1[0]
        dy = p2[1]-p1[1]
        pr = [p1[0] + dx * factor, p1[1] + dy * factor]
        return pr
    
    def getCoordinates(self):
        """
        Returns:
            the end points of the line in a list of coordinates format
            [xA,yA,xB,yB]
        """
        return [self.pA[0],self.pA[1],self.pB[0],self.pB[1]]        
    
    def getSamplePoints(self,n_points = 10):
        """
        Returns:
            np.array contaning n_points equally spread along the line
        Note:
            n_points >= 2
            The two ends of the line are always included,
            just exclude the first and last entry if you are not interested in them.
        """
        # adding one end of the line
        points = np.array([self.pA])
        # adding the points in between
        factor = 1/n_points
        if factor < 0.5:
            for d in np.arange(factor,1,factor):
                p = self.getPointBetween(self.pA,self.pB,d)
                points = np.append(points,[p],axis=0)   
        # adding the second end of the line
        points = np.append(points,[self.pB],axis=0)
        return points
                

    def __str__(self):
        str =  "pA = {}, pB = {}, len = {}, arc2 = {:.2f}";
        str = str.format(self.pA,self.pB,self.cm,self.length,
                         np.rad2deg(self.arc4),np.rad2deg(self.arc2))
        return str
    
    def __repr__(self):
        return self.__str__()
    
    
def enrichLines(lines):
    """
    Args:
        hlines: list of lines denoted by the ending point choordinates
        
    Returns:
        a list of lines expressed as Line2D objects
    """
    l = []
    for line in lines:
        for x1,y1,x2,y2 in line: 
            l.append(Line2D.fromAB(x1,y1,x2,y2))
    return l

def splitIntoCoordinates(points):
    """
    split the array of points into two arrays, one for each coordinate
    """
    r = np.hsplit(points,np.array([1,2]))
    x = r[0].reshape(len(points))
    y = r[1].reshape(len(points))
    return x,y

def mergeIntoPoints(xs,ys):
    """
    merge two arrays of coordinates into an array of points
    """
    points = np.empty((0,2),dtype=xs.dtype)
    for i,x in enumerate(xs):
        points = np.append(points,[[x,ys[i]]],axis=0)
    return points

def computeRegression(x,y,a,b,n=30):
    """
    Args:
        x: list of x-axis coordinates of the given points
        y: list of x-axis coordinates of the given points
        a: lower limit along the x-axis from wich you want to compute the 
           regression
        b: upper limit alogn the x-axis up to which you want to compute the
           regression
        n: number of points you want to compute evenly spread in [a,b] range

    Returns:
        x_points:  x-axis coordinates of evenly spread points in [a,b] range
        reg_model: function to call to get the y-axis value. 
                   e.g. reg_model(3) gives the value of the computed
                   regression model at point x=3 

    """
    reg_model = np.poly1d(np.polyfit(x, y, 1))
    x_points = np.linspace(a, b, n)
    return x_points,reg_model

def computeRegressionPoints(points,x_span,n=30):
    """
    Args:
        x_span: range of the independent variable of the regression: 
                e.g. (0,100)
        n:      amount of points you want to compute
                
    Returns:
        a numpy.array of `n` points representing the regression model
        within `x_span` range.
    """
    px,py = splitIntoCoordinates(points)
    xs,model = computeRegression(px,py,x_span[0],x_span[1],n)
    ys = [model(x) for x in xs]
    return mergeIntoPoints(xs,ys)