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
        #polynomial coefficients averaged over the last n iterations
        self.best_fit_left = None
        self.best_fit_right = None
        #polynomial coefficients for the most recent fit
        self.cur_fit_left = None
        self.cur_fit_right = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x,y values for detected line pixels, left
        self.line_pixels_left = None
        self.line_pixels_right = None
    

class LaneLineTracker(Line):
    def __init__(self):
        Line.__init__(self)
        self.camCal_mtx = None
        self.camCal_dist = None
        self.THRES_MIN_VALID_THRES = 100.0
        self.THRES_MIN_VALID_WIN_NUM = 15
        self.THRES_MAX_CUR_RAD_DELTA_MET = 500
        self.THRES_MAX_INVALID_CNT = 3
        self.THRES_MAX_LANE_WIDTH = 900
        self.X_MET_PER_PIX = 3.7/700.0
        self.Y_MET_PER_PIX = 30.0/720.0
        self.bIsCurLineValid = False
        self.num_of_invalid_cnt = 0

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
    def runMagSobel(self, img, sobelKernel=3, thresh=(0, 255)):
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
        absGradDir = np.abs(np.arctan2(np.abs(sobelY), np.abs(sobelX)))
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
        IMG_SIZE = (1280,720)
        src = np.float32([[(200, IMG_SIZE[1]), (581, 460), (705, 460), (1130, IMG_SIZE[1])]])
        dst = np.float32([[(360, IMG_SIZE[1]), (360, 0), (980, 0), (980, IMG_SIZE[1])]])
        #src = np.float32([[(270, 685), (612, 440), (671, 440), (1066, 685)]])
        #dst = np.float32([[(360, 720), (360, 0), (980, 0), (980, 720)]])
        transMtx = cv2.getPerspectiveTransform(src, dst)
        invTransMtx = cv2.getPerspectiveTransform(dst, src)
        return transMtx, invTransMtx

    def findLaneLines(self, warped, window_width, window_height, margin):


        window_left = []  # Store the (left,right) window centroid positions per level
        window_right = []
        idx_y = warped.shape[0]
        window = np.ones(window_width)  # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        if self.detected:
            l_center = self.line_pixels_left[-1]
            r_center = self.line_pixels_right[-1]
        else:
            # Sum quarter bottom of image to get slice, could use a different ratio
            l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
            l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
            r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
            r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

        # Add what we found for the first layer
        idx_y -= (window_height / 2.0)
        window_left.append((l_center,idx_y))
        window_right.append((r_center,idx_y))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(warped.shape[0] / window_height)):
            # convolve te window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width / 2

            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
            l_peak_idx = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index

            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
            r_peak_idx = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index

            idx_y -= window_height
            # Validate if the convolution peak is larger than the threshold for better confidence
            if conv_signal[l_peak_idx] > self.THRES_MIN_VALID_THRES:
                l_center = l_peak_idx - offset
                window_left.append((l_center,idx_y))
            if conv_signal[r_peak_idx] > self.THRES_MIN_VALID_THRES:
                r_center = r_peak_idx - offset
                window_right.append((r_center,idx_y))

        return window_left, window_right

    def computeLaneCurv(self, win_cent_left, win_cent_right):
        num_win_left = len(win_cent_left)
        num_win_right = len(win_cent_right)
        gain_left = float(num_win_left)/(num_win_left + num_win_right)
        ploty = np.linspace(0, 719, num=720)
        left_fit = np.polyfit(np.transpose(win_cent_left)[1], np.transpose(win_cent_left)[0], 2)
        right_fit = np.polyfit(np.transpose(win_cent_right)[1], np.transpose(win_cent_right)[0], 2)

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        y_eval = ploty[int(len(ploty)/2)]
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        #Weight based on the number of valid windows
        #To put more weight on the sold line than discrete line
        best_curv = gain_left*left_curverad + (1.0-gain_left)*right_curverad
        return left_fitx, right_fitx, best_curv*self.Y_MET_PER_PIX

    def window_mask(self, width, height, img_ref, center_width, center_height):
        output = np.zeros_like(img_ref)
        output[int(center_height-height/2.0):int(center_height+height/2.0), max(0, int(center_width - width / 2)):min(int(center_width + width / 2), img_ref.shape[1])] = 1
        return output
    def setCarPosInLane(self, x_down_left, x_down_right, img_width):
        cent_x_pos_img = img_width/2.0
        cent_x_pos_lane = (x_down_left + x_down_right)/2.0
        lane_x_offset = cent_x_pos_lane - cent_x_pos_img
        return lane_x_offset*self.X_MET_PER_PIX
    def processImage(self, img):
        img = self.undistImg(img)
        # Apply gradient threshold
        # abs sobel threshold (20, 100)
        binAbs = self.runAbsSobel(img, thresh=(30, 255))
        # mag sobel threshold (30, 100)
        binMag = self.runMagSobel(img, thresh=(30, 255))
        # dir sobel threshold (0.7, 1.5)
        binDir = self.runDirSobel(img, thresh=(0.7, 1.3))
        # HLS selection threshold (170, 255)
        binHls = self.runHlsSelect(img, thresh=(150, 255))
        binCombined = np.zeros_like(binAbs)
        #binCombined[(binAbs==1) & (binMag==1) & (binHls==1) & (binDir==1)] = 1
        binCombined[((binAbs==1)|(binHls==1)) & (binDir==1)] = 1
        # Apply Perspective transform
        imgSize = (img.shape[1],img.shape[0])
        transMtx, invTransMtx = self.getPerspectTransMtx(binCombined)
        birdViewImg = cv2.warpPerspective(binCombined, transMtx, imgSize)
        # Detect lane lines
        window_width = int(imgSize[0]/30)
        window_height = int(imgSize[1]/30)
        window_cent_left, window_cent_right = self.findLaneLines(birdViewImg, window_width, window_height, 50)
        # Determine lane line curvature
        left_fitx, right_fitx, best_curv = self.computeLaneCurv(window_cent_left, window_cent_right)
        lane_offset_in_met = self.setCarPosInLane(left_fitx[-1], right_fitx[-1], imgSize[0])
        # New lane validation check
        bIsNumOfSampleGood = (len(window_cent_left) > self.THRES_MIN_VALID_WIN_NUM) and (len(window_cent_right) > self.THRES_MIN_VALID_WIN_NUM)
        bIsCurvConsistent = self.detected == False  or (np.abs(self.radius_of_curvature - best_curv) < self.THRES_MAX_CUR_RAD_DELTA_MET)
        bIsLaneWidthGood = np.abs(left_fitx[-1] - right_fitx[-1]) < self.THRES_MAX_LANE_WIDTH
        if bIsNumOfSampleGood and bIsCurvConsistent and bIsLaneWidthGood:
            self.bIsCurLineValid = True
        else:
            self.bIsCurLineValid = False
            self.num_of_invalid_cnt += 1

        if self.bIsCurLineValid:
            self.num_of_invalid_cnt = 0
            self.detected = True
            self.radius_of_curvature = best_curv
            self.line_base_pos = lane_offset_in_met
            self.line_pixels_left = left_fitx
            self.line_pixels_right = right_fitx

        if self.num_of_invalid_cnt > self.THRES_MAX_INVALID_CNT:
            self.detected = False

        # Drawing the lines
        # Create an image to draw the lines on
        ploty = np.linspace(0, 719, num=720)
        warp_zero = np.zeros_like(birdViewImg).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.line_pixels_left, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.line_pixels_right, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, invTransMtx, imgSize)
        # Combine the result with the original image
        imgWithLines = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        cv2.putText(imgWithLines, "Curvature = %.1f m, Vehicle pos = %.1f m" %(self.radius_of_curvature, self.line_base_pos), (350,100), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255))

        return imgWithLines


bTestWithImage = False
bTestWithVideo = True

otLineLineTrk = LaneLineTracker()

# Camera calibration
imgPathToCal = './camera_cal'
imgSample = './camera_cal/calibration1.jpg'
otLineLineTrk.calibrateCamera(imgPathToCal)


if bTestWithVideo:
    from moviepy.editor import VideoFileClip
    videoToTest = 'project_video.mp4'
    videoToOut = 'project_video_with_lane.mp4'
    clip = VideoFileClip(videoToTest)
    clip_out = clip.fl_image(otLineLineTrk.processImage)
    clip_out.write_videofile(videoToOut, audio=False)


if bTestWithImage:
    imgPathToTest = './test_images'
    imgListToTest = os.listdir(imgPathToTest)
    for fFile in imgListToTest:
        img_org = cv2.imread(imgPathToTest+os.sep+fFile)
        # 2 Distortion correction
        img = otLineLineTrk.undistImg(img_org)
        # 3 Apply gradient threshold
        # abs sobel threshold (20, 100)
        binAbs = otLineLineTrk.runAbsSobel(img, thresh=(30, 255))
        # mag sobel threshold (30, 100)
        binMag = otLineLineTrk.runMagSobel(img, thresh=(30, 255))
        # dir sobel threshold (0.7, 1.5)
        binDir = otLineLineTrk.runDirSobel(img, thresh=(0.7, 1.3))
        # 4 Apply HLS selection threshold (170, 255)
        binHls = otLineLineTrk.runHlsSelect(img, thresh=(150, 255))
        binCombined = np.zeros_like(binAbs)
        #binCombined[(binAbs==1) & (binMag==1) & (binHls==1) & (binDir==1)] = 1
        binCombined[((binAbs==1)|(binHls==1)) & (binDir==1)] = 1
        # 5 Apply Perspective transform
        imgSize = (img.shape[1],img.shape[0])
        transMtx, invTransMtx = otLineLineTrk.getPerspectTransMtx(binCombined)
        birdViewImg = cv2.warpPerspective(binCombined, transMtx, imgSize)
        # 6 Detect lane lines
        window_width = int(imgSize[0]/30)
        window_height = int(imgSize[1]/30)
        window_cent_left, window_cent_right = otLineLineTrk.findLaneLines(birdViewImg, window_width, window_height, 50)
        # If we found any window centers
        if len(window_cent_left) > 0:
            # Points used to draw all the left and right windows
            l_points = np.zeros_like(birdViewImg)
            r_points = np.zeros_like(birdViewImg)

            # Go through each level and draw the windows
            for level in range(0, len(window_cent_left)):
                l_mask = otLineLineTrk.window_mask(window_width, window_height, birdViewImg, window_cent_left[level][0], window_cent_left[level][1])
                l_points[(l_points == 255) | ((l_mask == 1))] = 255

            # Go through each level and draw the windows
            for level in range(0, len(window_cent_right)):
                r_mask = otLineLineTrk.window_mask(window_width, window_height, birdViewImg, window_cent_right[level][0], window_cent_right[level][1])
                r_points[(r_points == 255) | ((r_mask == 1))] = 255

            # Draw the results
            template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
            zero_channel = np.zeros_like(template)  # create a zero color channel
            template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
            warpage = np.array(cv2.merge((birdViewImg, birdViewImg, birdViewImg)), np.uint8)  # making the original road pixels 3 color channels
            bwPipeLines = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results
            warpPipeLines = cv2.warpPerspective(bwPipeLines,invTransMtx,imgSize)

            # 7 Determine lane line curvature
            left_fitx, right_fitx, best_curv = otLineLineTrk.computeLaneCurv(window_cent_left, window_cent_right)
            lane_offset_in_met = otLineLineTrk.setCarPosInLane(left_fitx[-1], right_fitx[-1], imgSize[0])
            # Drawing the lines
            # Create an image to draw the lines on
            ploty = np.linspace(0, 719, num=720)
            warp_zero = np.zeros_like(birdViewImg).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(color_warp, invTransMtx, imgSize)
            # Combine the result with the original image
            imgWithLines = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        print("end")



