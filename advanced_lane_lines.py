#advanced_lane_lines.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os



# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
    

class LaneLineTracker():
    def __init__(self):
        self.camCal_mtx = None
        self.camCal_dist = None
        return
    def calibrateCamera(self, imgFolderPath):
        """
        Calibrate camera with the given images to measure the cal matrix
        :param imgFolderPath:
        :return: bool
        """
        CHESSBOARD_SIZE = (9,6)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = os.listdir(imgFolderPath)

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(imgFolderPath+os.sep+fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                #cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
                #write_name = 'corners_found'+str(idx)+'.jpg'
                #cv2.imwrite(write_name, img)
                #cv2.imshow('img', img)
                #cv2.waitKey(500)
        img_size = gray.shape
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        if ret:
            self.camCal_dist = dist
            self.camCal_mtx = mtx
        return ret
        
    def undistImg(self, img):
        """
        :param img:
        :return undistored img:
        """
        return cv2.undistort(img, self.camCal_mtx, self.camCal_dist)
    def runHlsSelect(self, img, thresh=(0,255)):
        """
        Applying S channel selection in HLS color
        :param img:
        :param thresh:
        :return img binary:
        """
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        sChannel = hls[:,:,2]
        np.float64(sChannel)
        binOut = np.zeros_like(sChannel)
        binOut[(sChannel >= thresh[0]) & (sChannel <= thresh[1])] = 1
        return binOut
    def runAbsSobel(self, img, orient='x', sobelKernel=3, thresh=(0, 255)):
        """
        Run absolute sobel gradient selection
        :param self:
        :param img:
        :param orient:
        :param thresh:
        :return img binary:
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if orient == 'x':
            absSobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobelKernel))
        if orient == 'y':
            absSobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobelKernel))
        scaleFactor = 255/np.max(absSobel)
        absSobel = scaleFactor*absSobel
        binOut = np.zeros_like(absSobel)
        binOut[(absSobel >= thresh[0]) & (absSobel <= thresh[1])] = 1
        return binOut
    def runMagSobel(self, sobelKernel=3, thresh=(0, 255)):
        """
        Run magnitude sobel gradient selection
        :param self:
        :param sobelKernel:
        :param thresh:
        :return img binary:
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobelKernel)
        sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobelKernel)
        sobelMag = np.sqrt(sobelX**2 + sobelY**2)
        scaleFactor = 255/np.max(sobelMag)
        sobelMag = sobelMag*scaleFactor
        binOut = np.zeros_like(sobelMag)
        binOut[(sobelMag >= thresh[0]) & (sobelMag <= thresh[1])] = 1
        return binOut
    def runDirSobel(self, img, sobelKernel=3, thresh=(0, np.pi/2.0)):
        """
        Run directional sobel gradient selection
        :param self:
        :param img:
        :param sobelKernel:
        :param thresh:
        :return img binary:
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobelKernel)
        sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobelKernel)
        absGradDir = np.arctan2(np.abs(sobelY), np.abs(sobelX))
        binOut = np.zeros_like(absGradDir)
        binOut[(absGradDir >= thresh[0]) & (absGradDir <= thresh[1])] = 1
        return binOut
    def getPerspectTransMtx(self, img):
        """
        Transform img with perspective view
        :param self:
        :param img:
        :return img corrected:
        """
        offset = 100
        imgSize = (img.shape[1],img.shape[0])
        #TODO Find corners for the input of the perspective transform
        src = np.zeros((4,2), np.float32)
        dst = np.zeros((4,2), np.float32)
        src[0] = [500,300] # left up
        src[1] = [700,300] # right up
        src[2] = [720,1100] # right bottom
        src[1] = [720,200] # left bottom
        dst[0] = [offset,offset]
        dst[1] = [imgSize[0]-offset,offset]
        dst[2] = [imgSize[0]-offset,imgSize[1]-offset]
        dst[3] = [offset,imgSize[1]-offset]
        transMtx = cv2.getPerspectiveTransform(src, dst)
        invTransMtx = cv2.getPerspectiveTransform(dst, src)
        return transMtx, invTransMtx
    def findLaneLines(self, warped, window_width, window_height, margin):

        window_centroids = []  # Store the (left,right) window centroid positions per level
        window = np.ones(window_width)  # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(warped.shape[0] / window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(
                warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height),
                :], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            # Add what we found for that layer
            window_centroids.append((l_center, r_center))

        return window_centroids

    def computeLaneCurv(self):
        return NotImplemented

        

imgPathToCal = './camera_cal'
imgPathToTest = './test_images'
imgSample = './camera_cal/calibration1.jpg'
imgListToTest = os.listdir(imgPathToTest)
otLineLineTrk = LaneLineTracker()
#1 Camera calibration
otLineLineTrk.calibrateCamera(imgPathToCal)

for fFile in imgListToTest:
    img = cv2.imread(imgPathToTest+os.sep+fFile)
    #2 Distortion correction
    img = otLineLineTrk.undistImg(img)
    #3 Apply gradient threshold
    # abs sobel threshold (20, 100)
    binAbs = otLineLineTrk.runAbsSobel(img, thresh=(20, 100))
    # mag sobel threshold (30, 100)
    binMag = otLineLineTrk.runMagSobel(img, thresh=(30, 100))
    # dir sobel threshold (0.7, 1.5)
    binDir = otLineLineTrk.runDirSobel(img, thresh=(0.7, 1.5))
    #4 Apply HLS selection threshold (170, 255)
    binHls = otLineLineTrk.runHlsSelect(img, thresh=(170, 255))
    binCombined = np.zeros_like(binAbs)
    binCombined[(binAbs==1) | (binHls==1) & (binDir==1)] = 1
    #5 Apply Perspective transform
    imgSize = (img.shape[1],img.shape[0])
    transMtx, invTransMtx = otLineLineTrk.getPerspectTransMtx(binCombined)
    birdViewImg = cv2.warpPerspective(binCombined, transMtx, imgSize)
    #6 Detect lane lines
    #7 Determine lane line curvature
