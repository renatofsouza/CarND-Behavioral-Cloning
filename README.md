# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nVidia_model.png "Model Visualization"
[image2]: ./examples/center_2018_12_09_17_34_08_167.jpg "example 1"
[image3]: ./examples/center_2018_12_09_17_33_05_974.jpg "example 2"
[image4]: ./examples/VideoSample.png "Video output"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I initially stared my architecture with a very basic neural network containing a couple of layers just to validate that it could process my data and save the model. Then I tried out a couple of well known architectures. First, I started with a LeNet convolutional neural network architecture. Then I decided to switch to the nVidia model. The diagram below describes the nVidia architecture.  

![nVidia Architecture][image1]

I used Keras Lamba function to include image normalization with three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers. I added RELU activation functions on each fully-connected layer. The final layer (depicted as "output" in the diagram) is a fully-connected layer with a single neuron.

#### 2. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. 

I used validation_split=0.2 and epochs=6 

#### 3. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. My dataset consists of 2 to 3 laps around the track plus the data initialy provided by Udacity. 

In my first attempt, I recorded 1 lap around the track. I verified that the car was going off-road in some instances when driving autonomusly. 

During the first data recording session I had several instances where I was unable to keep the car in the center of the lane. I decided that my data was not good and deided to compare how the car would drive if I only used the data provided by Udacity. 

Once I trained my neural network with the Udacity data, the car drove much better. But it was crashing at some point near the bridge. 

I then decided to record additional data and combine with the Udacity data. This time, I recorded 2 to 3 laps around the track mostly driving as close as possible to the center lane with a few scenarios of recovery from the sidelines. 

Below are some sample images.

![Example 1][image2]
![Example 2][image3]


#### 4. Preprocess Data

I created a function to preprocess the data. My pre-process function basically executes the following steps:
- Crops the image to remove the sky and hood of the car as the neural network does not need to learn about these features. 
- Scale the image to 66x200x3 as this is the same input as the nVidia architecture.
- convert to YUV color space (as nVidia paper suggests)


#### 5. Data Augmentation and Data Filtering
During the process of loading the data in the read_data function I discard any data with speed <> 0.1 as this does not really imply any driving behavior. 

Then I add +0.25 and -0.25 to the angle measurement of the left and right cameras. 

Lastly, I augment the data by fliping horizontally and inverting steering angle of every image where the magnitude of the steering angle  is > 0.33. This process helps remove bias since the track mostly has turns to one side. 

### Video Output

Below is a link to the video outpot of my car being driven by the neural network model. 

[![Video Output][image4]](https://youtu.be/NHcDFnmteXQ)
