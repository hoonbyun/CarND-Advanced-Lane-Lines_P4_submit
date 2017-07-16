#advanced_lane_lines.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob



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
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(imgFolderPath)

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (8,6), corners, ret)
                #write_name = 'corners_found'+str(idx)+'.jpg'
                #cv2.imwrite(write_name, img)
                cv2.imshow('img', img)
                cv2.waitKey(500)
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
        binOut = np.zeros_like(sChannel)
        binOut[(sChannel >= thresh[0]) & (sChannel <= thresh[1])] = 1
        return binOut:
    def runAbsSobel(self, img, orient='x', thresh=(0, 255)):
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
            absSobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            absSobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        scaledSobel = np.unit8(255*absSobel/np.max(absSobel))
        binOut = np.zeros_like(scaledSobel)
        binOut[(scaledSobel >= thresh[0]) & (scaledSobel <= thresh[1])] = 1
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
        scaleFactor = 255.0/np.max(sobelMag)
        sobelMag = (sobelMag*scaleFactor).astype(np.unit8)
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
        binOut [(absGradDir >= thresh[0]) & absGradDir <= thresh[1]] = 1
        return binOut
    def runPerspectTrans(self, img):
        """
        Transform img with perspective view
        :param self:
        :param img:
        :return img corrected:
        """
        offset = 100
        #TODO Find corners for the input of the perspective transform
        src = np.float32()
        dst = np.float32()
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, img.shape)
    def findLaneLines(self):
        return NotImplemented:
    def computeLaneCurv(self):
        return NotImplemented:

        
def main():
    otLineLineTrk = LaneLineTracker()
    #1 Camera calibration
    otLineLineTrk.calibrateCamera(calImgPath)
    for img in imgs:
        #2 Distortion correction
        #3 Apply gradient threshold
        # abs sobel threshold (20, 100)
        # mag sobel threshold (30, 100)
        # dir sobel threshold (0.7, 1.5)
        #4 Apply HLS selection threshold (170, 255)
        #5 Apply Perspective transform
        #6 Detect lane lines
        #7 Determine lane line curvature
    
    
