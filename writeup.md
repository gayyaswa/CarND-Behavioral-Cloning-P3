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

I implemented the network using Keras by following the lecture and using the architecture image from NVIDIA for reference here published here https://devblogs.nvidia.com/deep-learning-self-driving-cars/.

#### 2. Attempts to reduce overfitting in the model

I tried implementing without the dropout layer and managed to succeed a lap also the validation loss is in closer range in comparison to training loss. But still try adding a dropout layer in the first fully connected layer and observed a slight decrease in the validation loss and the vehicle was still completing the lap in autonomous mode. Based on this I think overfitting isn't a concern for the dataset used in this project

#### 3. Model parameter tuning

The final model used an adam optimizer tried model with default learning rate which is -01 and worked well. During training I did tried to decreasing the learning rate -001 but didn't help much.

#### 4. Appropriate training data

Using the center image provided with data set was able to steer the vehicle straight also collected the data from simulator for steer left and right recovery which did help the vehicle steer left and right. But the steering wasn't smooth it did recover from few crashes so to smoothen the steering behavior augumented the steering angles by +0.15 and -0.15 for left and right images respectively

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
* As suggested in the lecture tried simple neural network to familiar with the process of training and running the network model using the simulator. 
* I tried implementing the Lenet from previous exercise and observed the vehicle was going straight
* I decided upon the NVIDIA CNN as it is complex with various layer and more nodes tested in real world to extract the features and act as a steering controller.


#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
