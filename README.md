# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You can submit your writeup in markdown or use another method and submit a pdf instead.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).


## Comparison with a Deep Neural Network

![Comparison][./output_images/cv-l2-4.1.001.png]

A traditional Computer Vision pipeline for object detection typically consists of separate processing stages, for instance, feature extraction, spatial sampling and classification. Different algorithms and techniques can be applied for these stages, and the parameters for each stage are usually tuned by hand.

In comparison, a Deep Neural Network designed for object detection can perform these tasks using a complex interconnected architecture where the stages are not as distinct. Lower layers of a neural network, i.e. the ones closer to the input data, typically perform some equivalent of feature extraction, while higher layers can localize and classify simultaneously to produce detections.

Both approaches have their strengths and weaknesses, in terms of performance, development effort, interpretability, etc. For a more in-depth discussion of how Deep Learning is being applied to object detection tasks, you may refer to the following papers (and references therein):

Redmon et al., 2015. You Only Look Once: Unified, Real-Time Object Detection. [arXiv]

W. Liu et al., 2015. SSD: Single Shot MultiBox Detector. [arXiv]

S. Ren et al., 2015. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. [arXiv]


-----------------------

## Writeup Vehicule Detection and Tracking
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/trainset.png
[image2]: ./output_images/Hogimages.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/example1.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.gif

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Data Set choice

**I use project data set and udacity dataset.**

From Udacity dataset, I use car tagged box to extract car images. And I use pedestrian tagged box, and I cut all images without vehicules to extract NotCar images.

Because on Udacity dataset, Cars are oriented in the same way, I flip all images of car to generalize it.

In order to avoid overfitting, I choose to buid my train set and test set like below:

**Train Cars Set** -> car project set (8792 samples) + cars udactity (4000 samples) + flipped image of cars udacity (4000 samples)

**Train NotCars Set** -> NotCar project set (8968 samples) + pedestrian udacity (2500 sapmples) + not cars udacity (5500 samples)

**Test Cars Set** -> cars udactity (5000 samples) + flipped image of cars udacity (5000 samples)

**Test Notcars Set** -> pedestrian udacity (3174 sapmples) + not cars udacity (7500 samples)

Also, I obtain independant sets like this:

* lenght of cars_train: 16792
* lenght of notcars_train: 16968
* lenght of cars_test: 10000
* lenght of notcars_test: 10674

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 16th to 19th code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=7`, `pixels_per_cell=(4, 4)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and i observe accurancy result of a linear SVC model for choosing parameters.

Finally, my final choice is to use together the 3 kind of features with parameters :
 * **HOG features**: color_space = 'HSV', orient = 7, pix_per_cell = 16, cell_per_block = 2
 * **Binned color features**: color_space = 'HSV', spatial_size = (16, 16)
 * **Color Histogram features**: color_space = 'HSV', hist_bins = 32
 
Also, I get 1116 features.

Then, I use forest of trees to detect and keep only the most important features. I use command `SelectFromModel(tree, prefit=True)` to do that.

Finally, I get 230 features.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the 20th to 23th code cell of the IPython notebook.

I trained different models:
 * Linear SVM
 * SVM kernel: RBF
 * Forest ensemble : ExtraTreesClassifier
 * Adaboost
 
I use GridsearchCV to optimize it.

Finally by regarding accuracy result, time to predict, and behavior on test images, I choose the **RBF kernel SVM** like the best model.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the 24th to 26th code cell of the IPython notebook.

I decided to search some window positions and some scales like below:
 * ystart = 416, ystop = 480, scale = 1., clf, cells_per_step = 1
 * ystart = 400, ystop = 600, scale = 2., clf, cells_per_step = 2
 * ystart = 480, ystop = 720, scale = 3., clf, cells_per_step = 2

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using HSB 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video.mp4)
![video][./output_project_video.gif]

I try to perform my pipeline on other challenge video of project 4.

Here's a [link to challenge video result](./output_images/output_challenge_video.mp4)

Here's a [link to hardest challenge video result](./output_images/output_hchallenge_video.mp4)

Then I perform on the project video the pipelines of project 4 and project 5 together.

Here's a [link to my video result with vehicles and Lane detection](./output_images/output_project2_video.mp4)



#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in the 27th to 29th code cell of the IPython notebook.

By annalizing, that car detection which appears and disappears quickly like a mirage are some false positive. I use a story of heatmap by using an image object that save the previous detection.

Also, I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and I add +1 for each detection to the previous detection. From the negative detection I substract -1 to the previous heatmap detection. Also, I smooth appeared and disappeared car detection. And then I threshold the heatmap object to identify vehicle positions and delete false positive.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's the result of test video showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

[link to my test video result](./output_images/output_test_video.mp4)



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project, I have some troubles with quality of my learning model. I have many false positives. I find to use one history of detection to smooth detection by considering that cars who appears only during a short period is a false poitive.

In a second time, It was difficult to build a better dataset from Udacity set. So, I consider that a better data sample set could be find to improve result.

And maybe my feature selection could be improve too.

I remark that to detect white car on clear color road are difficult. And there are many false positives in a shadow image. 

