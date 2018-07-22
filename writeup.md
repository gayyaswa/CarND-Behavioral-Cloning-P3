# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./nvidia-cnn-architecture.png "NVIDIA CNN Architecture"
[image2]: ./examples/steer_left/left_2018_07_20_00_14_58_195.png "steer left1 recovery"
[image3]: ./examples/steer_left/left_2018_07_20_00_14_59_050.jpg "steer left2 recovery"
[image4]: ./examples/steer_left/left_2018_07_20_00_15_01_045.jpg "steer left3 recovery"
[image5]: ./examples/steer_left/left_2018_07_20_00_15_02_159.jpg "steer left4 recovery"
[image6]: ./examples/steer_right/right_2018_07_20_00_29_29_197.jpg "steer right1 recovery"
[image7]: ./examples/steer_right/right_2018_07_20_00_29_29_476.jpg "steer right2 recovery"
[image8]: ./examples/steer_right/right_2018_07_20_00_29_30_062.jpg "steer right3 recovery"
[image9]: ./examples/steer_right/right_2018_07_20_00_29_31_189.jpg "steer right4 recovery"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 contains a recorded lap in autonomous mode
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
The submission includes model.h5 which can be run in autonomous mode with the script drive.py provided part of the repo 

#### 3. Submission code is usable and readable

The model.py checked in part of repo contains the CNN implementation based on the NVIDIA architecture.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I implemented the NVIDIA CNN using Keras by following the lecture and here is the reference for the network https://devblogs.nvidia.com/deep-learning-self-driving-cars/.

#### 2. Attempts to reduce overfitting in the model

I tried implementing without the dropout layer and managed to succeed a lap also the validation loss is in closer range in comparison to training loss. But still try adding a dropout layer in the first fully connected layer and observed a slight decrease in the validation loss and the vehicle was still completing the lap in autonomous mode. Based on this I think overfitting isn't a concern for the dataset used in this project

#### 3. Model parameter tuning

The final model used an adam optimizer tried model with default learning rate which is .01 and worked well. During training I did tried to decreasing the learning rate .001 but didn't help much.

#### 4. Appropriate training data

Using the center images provided with data set was able to steer the vehicle straight also collected the data from simulator by simulating steer left and right recovery which did help the vehicle steer left and right. But the steering wasn't smooth eventhough it did recover from few crashes, so to smoothen the steering behavior augumented the steering angles by +0.15 and -0.15 for left and right images respectively. Overall was able to train the vehicle with 40000 samples and finish the lap.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
* As suggested in the lecture tried simple neural network to familiar with the process of training and running the network model using the simulator. 
* I tried implementing the Lenet from previous exercise and observed the vehicle was going straight
* I decided upon implementing the NVIDIA CNN as the final solution because of its complexity and more number of nodes. Also the network architecture was tested in real world to extract the road features and steer the vehicle successfuly


#### 2. Final Model Architecture

The final model architecture (model.py lines 97-110) consisted of a convolution neural network with the following layers and layer sizes. Added a final ouput layer of single node so that the network will be trained to output one steering control angle.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

##### Image Preprocessing

* The camera images are cropped to remove the hood of the car, moutains, lakes and trees by using the top and bottom parameter. Cropping is done part of the model so it can be done using GPU to acheive better performance

* As part of preprocessing the images are normalized and mean centereed to 0.5 which helped reducing the error and improve the training loss

* The images from the dataset are split between 80% and 20% for training and validation respectively

##### Steps followed to capture the driving behaviour

* Center images from the provided data set is used to train the network and was able to acheive the straight steer training behavior
but it crashed in the left turn near.
* To train with steer left behaviour captured several recovery images across the track and ended up collecting approximately 7000 samples refer to sample images below
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
* To train with steer right behviour drove the vehicle in clockwise direction and captured several recovery images across the track and collected around 6500 samples refer to sample images below
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

###### Data Augumentation and smoothing steering behaviour

* Even with the capture images from the above step still vehicle went off the road markers in few occassions along the track
* So decided to use the left, right camera images from original dataset together with the capture images resulted in 40000 sample approxiamtely and did help learn the steering behavior successfuly
* For each right and legt images from sample augumented the steering angle by -0.15 and +0.15 respectively which resulted in a smooth steering behavior and also ended up the vehicle staying within the lane markers throughout the track.

###### Challenges 

* As per the discussion in our community tried training the vehicle with fewer samples for steering left and right but fewer samples weren't enough to learn the steering behavior.
* Due to fewer samples adding few additional data altered the vehicle steering behavior a lot but having more samples helped the network learn the steering behavior in generalized way


As mentioned in the above section the data is split between 80% and 20% of training and validation respectively. Also modifying the default learning rate of adam optimizer to 0.001 didn't help in achieving the desired steering behvior

