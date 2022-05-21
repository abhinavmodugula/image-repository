"""
Defines the model used to search images
by various metrics.

Current Features:
1) General Image similarity

A feature must implement a describe() and
compare() method. The descrive method returns an 
array representing some feature about an image. The 
compare() method compares two of those features and
returns a score of how different they are.

The search method will display images with low distance
defined by the feature's compare method

"""
import numpy as np
from enum import Enum
import imutils
import cv2

class FeatureType(Enum):
    GENERAL = 1

class FeatureFactory:
    def create(type: FeatureType):
        if type == FeatureType.GENERAL:
            return ImageSimilarityFeature()
        else:
            raise "Unknown feature"


class Feature:
    pass

class ImageSimilarityFeature(Feature):
    def __init__(self, bins=100):
        self.bins = bins
        self.name = "general"
    
    """ Computes feature vec for image
        Parameters:
        ---------------
        image   image array (x,y,3)

        Returns:
        ---------------
        featrues    numpy array
    """
    def describe(self, image):
        # convert the image to the HSV color space and initialize
        # the features used to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        
        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        # divide the image into four rectangles/segments (top-left,
        # top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
        (0, cX, cY, h)]
        # construct an elliptical mask representing the center of the
        # image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        
        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)
            # extract a color histogram from the image, then update the
            # feature vector
            hist = self._histogram(image, cornerMask)
            features.extend(hist)
        
        # extract a color histogram from the elliptical region and
        # update the feature vector
        hist = self._histogram(image, ellipMask)
        features.extend(hist)
        # return the feature vector
        return features
    
    def _histogram(self, image, mask):
        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel
        hist = cv2.calcHist([image.astype('uint8')], [0, 1, 2], mask, [self.bins, self.bins, self.bins], [0, 180, 0, 256, 0, 256])
        #hist = cv2.calcHist([img.astype('uint8')],[0],None,[256],[0, 256]) 
        
        # normalize the histogram if we are using OpenCV 2.4
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()
        # otherwise handle for OpenCV 3+
        else:
            hist = cv2.normalize(hist, hist).flatten()
        # return the histogram
        return hist
    
    """ Compares two feature vectors
        Parameters:
        ---------------
        feature1    numpy array (N,M)
        feature2    numpy array (N,M)

        Returns:
        ---------------
        distance    int
    """
    def compare(self, feature1, feature2):
        return self._distance(feature1, feature2)
    
    def _distance(self, A, B):
        return np.linalg.norm(A-B)
    
    def _chi2_distance(self, histA, histB, eps = 1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
        # return the chi-squared distance
        return d