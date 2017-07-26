## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/test_before_cal.PNG "Camera calibration"
[image1-1]: ./output_images/test_after_cal.PNG "Camera calibration"
[image2]: ./output_images/test_distor_cor.PNG "Distortion corrected"
[image3]: ./output_images/test_bin_abs.PNG "Binary Example"
[image4]: ./output_images/test_perspective_birdview.PNG "Warp Example"
[image5]: ./output_images/test_fit_win.PNG "Fit Visual"
[image6]: ./output_images/test_warp_back_final.PNG "Output"
[video1]: ./project_video_with_lane.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

It is done in the first step in the function "processImage()" where all necessary image processing takes place.
The given chess board images are used for the camera calibration, the "findChessboardCorners()" method used to detect the corner of the board.
All known 3d points are stored in "objpoints" for the real world space while the detected corners are added in "imgpoints".
Those two lists are used as an input of the function "cv2.calibrateCamera()" to get the camera matrix.

Chessboard, original
![alt text][image1]

Chessboard, with the camera calibration  
![alt text][image1-1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
Various gradients methods are combined to get the best lane line in gray scale, the first gradient method is a x direction Sobel "runAbsSobel()" which gives the best lane line shape.
But it is bit noisy compared to the another gradient in HLS color transforms "runHlsSelect()", S channel only shows the most vivid lane line, the last gradient is a directional Sobel "runDirSobel()" to leave the vertical objects only.
```python
binCombined[((binAbs==1)|(binHls==1)) & (binDir==1)] = 1
```

Binary image after the gradients in gray 
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
Figuring out the right source and destination points were the most tricky job in this project to get the right perspective transform.
It is also very important to get the right bird view to detect the lines and measure the curvature.

```python
IMG_SIZE = (1280,720)
src = np.float32([[(200, IMG_SIZE[1]), (581, 460), (705, 460), (1130, IMG_SIZE[1])]])
dst = np.float32([[(360, IMG_SIZE[1]), (360, 0), (980, 0), (980, IMG_SIZE[1])]])
```

Bird view image
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
My function "findLaneLines()" to identify the lane line pixels uses the convolution method, the defined search window box with all white pixels in gray scale slides along the image out of the all gradients steps in gray scale and in the bird view.
It scans the target image range to find the peak in x axis, in that way the most white pixels scatters will be correlated with the window and find the x position.
From the discrete candidates pixels piling up along the lane line, those inputs are used to find the best fit poly coefficients in y axis as an input of the function since the change in x axis is small.

Found windows and fit lines derived from the poly coefficients
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

From the obtained poly fits from the previous step, it draws each lines with the given range of the y values based on the size of the image, the function "computeLaneCurv()" uses those input to estimate the radius of curvature of each line.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
The found lane lines are transformed back to the warp image onto the original image, below is the example with the detected lane line.
![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_with_lane.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* Discrete and thin lane line 
    * It is a challenge to detect the discrete lane lines on the right side compared to the solid lane line, for the radius curvature estimation, I combine to values from the lest and right with different weights based on the number of valid x pixels
```python
gain_left = float(num_win_left)/(num_win_left + num_win_right)
best_curv = gain_left*left_curverad + (1.0-gain_left)*right_curverad
```
    

* Wrong correlation peak found in convolution
    * Sometime the convolution returns wrong x pixel point where the peak is found
    I noticed that those cases, its peak value is smaller than the good one, applying the peak value threshold helps to avoid the wrong peak detection
```python
                # Validate if the convolution peak is larger than the threshold for better confidence
            if conv_signal[l_peak_idx] > self.THRES_MIN_VALID_THRES:
                l_center = l_peak_idx - offset
                window_left.append((l_center,idx_y))
            if conv_signal[r_peak_idx] > self.THRES_MIN_VALID_THRES:
                r_center = r_peak_idx - offset
                window_right.append((r_center,idx_y))
```
    
* Shadows or spots on the road
    * Shadow or spots on the road fools the pipe line detect algorithms, adding some validity checks helps to eliminate those epochs but propagate the previous one instead
```python
            # New lane validation check
        bIsNumOfSampleGood = (len(window_cent_left) > self.THRES_MIN_VALID_WIN_NUM) and (len(window_cent_right) > self.THRES_MIN_VALID_WIN_NUM)
        bIsCurvConsistent = self.detected == False  or (np.abs(self.radius_of_curvature - best_curv) < self.THRES_MAX_CUR_RAD_DELTA_MET)
        bIsLaneWidthGood = np.abs(left_fitx[-1] - right_fitx[-1]) < self.THRES_MAX_LANE_WIDTH
```