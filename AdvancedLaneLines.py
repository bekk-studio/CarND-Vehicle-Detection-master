
# coding: utf-8

# # Advanced Lane Lines master

# ## Used libraries

# In[1]:

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from skimage import img_as_ubyte
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from scipy.signal import gaussian
get_ipython().magic('matplotlib inline')


# ## Distortion correction
# Like in explanation. Nothing has been add.

# In[2]:

def cal_undistort(img):
    mtx = np.array([[1.15396093e+03, 0.00000000e+00, 6.69705357e+02], 
       [0.00000000e+00, 1.14802496e+03, 3.85656234e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([[-2.41017956e-01, -5.30721171e-02, -1.15810354e-03, -1.28318858e-04, 2.67125301e-02]])
    return cv2.undistort(img, mtx, dist, None, mtx)


# In[3]:

def abs_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient abs 
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1) 
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Apply threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    # Apply threshold
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return mag_binary

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely) 
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    # Apply threshold
    dir_binary = np.zeros_like(dir_grad)
    dir_binary[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    return dir_binary


def gradients_threshold(img, ksize=3, gradx_th=(30, 255), grady_th=(30, 255), mag_th=(30, 255), dir_th=(0.7, 1.3)): 

    # Apply each of the thresholding functions
    gradx_binary = abs_thresh(img, orient='x', sobel_kernel=ksize, thresh=gradx_th)
    grady_binary = abs_thresh(img, orient='y', sobel_kernel=ksize, thresh=grady_th)
    mag_binary = mag_thresh(img, sobel_kernel=ksize, thresh=mag_th)
    dir_binary = dir_thresh(img, sobel_kernel=ksize, thresh=dir_th)
    
    combined = np.zeros_like(dir_binary)
    combined[((gradx_binary == 1) & (grady_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    return combined

def threshold_pipeline(img, s_thresh=(120, 255), h_thresh=(0, 50)):
    
    img = np.copy(img)
    #clahe = equalize_adapthist(img).astype(np.float32)
    #clahe = img_as_ubyte(clahe)
    
    # Convert to HLS and HSV color space and separate the channels
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    v_channel = hsv[:,:,2]
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    
    # Threshold combined gradient of S channel with more accurate thresholds
    grads_binary = gradients_threshold(s_channel, gradx_th=(15, 255), grady_th=(15, 255), mag_th=(25, 255), dir_th=(0.7, 1.3))
    
    # Threshold combined gradient of V channel
    gradv_binary = gradients_threshold(v_channel)
    
    
    combined_binary = np.zeros_like(gradv_binary)
    combined_binary[((grads_binary == 1) | (gradv_binary == 1))] = 1    
    
    return img_as_ubyte(combined_binary)


# In[4]:


def perspective_transform(img):
    lane_width = 600
    img_size = img.shape[:2]
    border = (img_size[1] - lane_width)//2
    src = np.float32([[200,img_size[0]],[1120,img_size[0]],[723,470],[563,470]])
    dst = np.float32([[border, img_size[0]],[border + lane_width, img_size[0]], [border + lane_width, 0],[border, 0]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size[1::-1], flags=cv2.INTER_LINEAR)    
    # return warped and Matrix for later when we will inverse the transformation
    return M, warped


# ## Tracking
# class to receive the characteristics of each line detection

# In[5]:

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
        # image iteration
        self.iteration = 0


#    ## Detect lane lines
#    * If the first image or if last detection have less than 2000 pixels or each 10 images:
#        * Define a gaussian mask in order to favour line in the center during the bottom line detection center
#        * Define a gaussian window for line detection. Normal distribution is better to center it.
#        * By convolution, find the optimal fit
#    * else, detect line from the last fits with a margin of 60 pixels
#    * Smooth the fit with the 10th last fit with a learning rate of 0.9, then average with a ratio of 70%/30% between left and right lines.
#    * Display and select, pixel inside the windows
# 

# In[6]:

# To detect lane more accurately at the bottom of image, we define a mask which focus on the center of image


# In[7]:


# function for detection who prefer the center of image. Used only for choosing the bottom of line 
def center_focus(img, lane_width=600):
    focus_window = 2 * gaussian(img.shape[1], std=200) + 1
    new_img = []
    for row in img:
        new_img.append(row * focus_window)
    new_img = np.array(new_img)
    return new_img

# In[9]:

# selection window settings
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
           max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(warped, window_width=50, window_height=20, margin=40):
    
    # In order to maximize detection lane in center
    focus_warped = center_focus(warped)
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    # Create our window template that we will use for convolutions
    window = gaussian(151, std=20)
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Use len(window)/2 as offset because convolution signal reference is at right side of window, not center of window
    offset = len(window)/2
    
    # Sum half bottom of image to get slice
    l_sum = np.sum(focus_warped[int(focus_warped.shape[0]/2):,:int(focus_warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-offset
    r_sum = np.sum(focus_warped[int(focus_warped.shape[0]/2):,int(focus_warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-offset+int(focus_warped.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
        if np.max(conv_signal[l_min_index:l_max_index]) > 10000.: #if more about 50 pixels
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
        if np.max(conv_signal[r_min_index:r_max_index]) > 10000.: #if more about 50 pixels
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids

def display_window(warped, window_width=50, window_height=20):
    window_centroids = find_window_centroids(warped)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channle 
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        
    return output


# In[10]:

def find_lines(binary_warped, left, right, display=False, n=10, window_width=50, window_height=20):
    
    # Create an output image to draw on and  visualize the result
        
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    # Create empty lists to receive left and right lane pixels  by windows
    left_lane = []
    right_lane = []
    
    if left.detected == False and right.detected == False:
        window_centroids = find_window_centroids(binary_warped)
        
        if display == True:
            out_img = display_window(binary_warped)
        
        # Go through each level and draw the windows
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,binary_warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,binary_warped,window_centroids[level][1],level)
            # Identify the nonzero pixels in x and y within the window 
            l_points = l_mask * binary_warped
            r_points = r_mask * binary_warped

            # Append these points to the lists
            left_lane.append(l_points)
            right_lane.append(r_points)
    
        # Sum the arrays of pixels by window to obtain the whole pixels
        left_lane_all = np.sum(np.array(left_lane), axis=0)
        right_lane_all = np.sum(np.array(right_lane), axis=0)
        
        # Extract left and right line pixel positions
        left.allx = left_lane_all.nonzero()[1]
        left.ally = left_lane_all.nonzero()[0] 
        right.allx = right_lane_all.nonzero()[1]
        right.ally = right_lane_all.nonzero()[0]
        
    else:
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        find_margin = 60 #width of detection area
        left_lane_inds = ((nonzerox > (left.best_fit[0]*(nonzeroy**2) + left.best_fit[1]*nonzeroy + left.best_fit[2] - find_margin)) 
                        & (nonzerox < (left.best_fit[0]*(nonzeroy**2) + left.best_fit[1]*nonzeroy + left.best_fit[2] + find_margin))) 
        right_lane_inds = ((nonzerox > (right.best_fit[0]*(nonzeroy**2) + right.best_fit[1]*nonzeroy + right.best_fit[2] - find_margin)) 
                        & (nonzerox < (right.best_fit[0]*(nonzeroy**2) + right.best_fit[1]*nonzeroy + right.best_fit[2] + find_margin)))  

    
        # Again, extract left and right line pixel positions
        left.allx = nonzerox[left_lane_inds]
        left.ally = nonzeroy[left_lane_inds] 
        right.allx = nonzerox[right_lane_inds]
        right.ally = nonzeroy[right_lane_inds]
    
        if display == True:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            window_img = np.zeros_like(out_img)
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left.bestx-find_margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left.bestx+find_margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right.bestx-find_margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right.bestx+find_margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    
    if display == True:
        out_img[left.ally, left.allx] = [255, 0, 0]
        out_img[right.ally, right.allx] = [0, 0, 255]
    
    # Fit a second order polynomial to each
    if len(left.ally) > 0:
        left.current_fit = np.polyfit(left.ally, left.allx, 2)
    if len(right.ally) > 0:
        right.current_fit = np.polyfit(right.ally, right.allx, 2)
    
    # the last polynomial fit
    left.recent_xfitted.append(left.current_fit[0]*ploty**2 + left.current_fit[1]*ploty + left.current_fit[2])
    left.recent_xfitted = left.recent_xfitted[-n:]
    right.recent_xfitted.append(right.current_fit[0]*ploty**2 + right.current_fit[1]*ploty + right.current_fit[2])
    right.recent_xfitted = right.recent_xfitted[-n:]
    
    # average with a learning rate of 0.9
    left.bestx = np.average(np.array(left.recent_xfitted), axis=0, weights=[0.9**i for i in range(len(left.recent_xfitted))])
    right.bestx = np.average(np.array(right.recent_xfitted), axis=0, weights=[0.9**i for i in range(len(right.recent_xfitted))])
    # average with ration 70%/30% between left and right lines
    left.bestx = np.average([left.bestx, right.bestx + left.bestx[-1] - right.bestx[-1]],
                            axis=0, weights=[0.7, 0.3])
    right.bestx = np.average([right.bestx, left.bestx + right.bestx[-1] - left.bestx[-1]], 
                            axis=0, weights=[0.7, 0.3])
    
    left.best_fit = np.polyfit(ploty, left.bestx, 2)
    right.best_fit = np.polyfit(ploty, right.bestx, 2)
    
    # image iteration in the video
    left.iteration += 1
    right.iteration += 1
    
    # if there are less than 2000 fit pixels, Use convolution find policy
    if len(left.allx) < 2000 or len(right.allx) < 2000: 
        left.detected = False
        right.detected = False
    # else use the last fit polynom window to find pixels
    else:
        left.detected = True
        right.detected = True
    
    if display == True:
        return out_img


# ## Find the lane curvature

# In[11]:

def lane_curvature(img_ref, left, right):
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 22/img_ref.shape[0] # meters per pixel in y dimension
    xm_per_pix = 3.7/600 # meters per pixel in x dimension
    
    y_eval = float(len(img_ref)-1)
    A_left = left.best_fit[0]*xm_per_pix/(ym_per_pix**2)
    B_left = left.best_fit[1]*xm_per_pix/ym_per_pix
    A_right = right.best_fit[0]*xm_per_pix/(ym_per_pix**2)
    B_right = right.best_fit[1]*xm_per_pix/ym_per_pix
    
    left.radius_of_curvature = ((1 + (2*A_left*y_eval*ym_per_pix + B_left)**2)**1.5) / np.absolute(2*A_left)
    right.radius_of_curvature = ((1 + (2*A_right*y_eval*ym_per_pix + B_right)**2)**1.5) / np.absolute(2*A_right)
    # Now our radius of curvature is in meters
    


# ## Find offset of the lane center

# In[12]:

def offset(img_ref, left, right):
    # Define offset between vehicule center and lane center.  
    
    # Define conversions in x from pixels space to meters
    xm_per_pix = 3.7/600 # meters per pixel in x dimension
    
    y_eval = float(len(img_ref) - 1)
    
    x_bottom_left = left.best_fit[0]*((y_eval)**2) + left.best_fit[1]*y_eval + left.best_fit[2]
    x_bottom_right = right.best_fit[0]*((y_eval)**2) + right.best_fit[1]*y_eval + right.best_fit[2]
    
    img_center = img_ref.shape[1]/2
    
    left.line_base_pos = (img_center - x_bottom_left)*xm_per_pix
    right.line_base_pos = (img_center - x_bottom_right)*xm_per_pix

    


# ## Drawing lane
# the final pipeline

# In[13]:

def drawing_lane_pipeline(image, left, right):

    # undistortion
    undist = cal_undistort(image)
    # thresholding
    thresh = threshold_pipeline(undist)
    # Warped image
    M, warped = perspective_transform(thresh)
    # Detect Lines
    find_lines(warped, left, right)
    if left.iteration % 10 == 1: # To make more readable values. Update it every 10 pictures
        lane_curvature(warped, left, right)
        offset(warped, left, right)
    radius = np.mean([left.radius_of_curvature, right.radius_of_curvature])
    offsetm = (left.line_base_pos + right.line_base_pos)/2
    
    # Add on images Radius and offset impormation
    curvature_string = 'Radius of curvature of lane : {0:.0f}m'.format(radius)
    lane_center_offset_string = 'Offset with lane center: {0:.2f}m'.format(offsetm)
    
    cv2.putText(img=undist, text=curvature_string, org=(350,100), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
    cv2.putText(img=undist, text=lane_center_offset_string, org=(350,150), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left.bestx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right.bestx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = np.linalg.inv(M)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

