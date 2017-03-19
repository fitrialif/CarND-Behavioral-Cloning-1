# Behavioral Cloning [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This is my solution of Project 3 of Udacity's Self Driving Car Nanodegree.  

### Goals & steps of the project
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Project files
|filename|description
|---|---|
|[model.py](./model.py)|containing the script to create and train the model|
|[drive.py](./drive.py)|for driving the car in autonomous mode|
|[model.h5](./model.h5)|containing a trained convolution neural network|
|[video.mp4](./video.mp4)|video recording in autonomous mode

### Video

[![Autonomous driving with 30 mph](http://img.youtube.com/vi/JexvhXDvB90/maxresdefault.jpg)](http://www.youtube.com/watch?v=JexvhXDvB90 "Autonomous driving with 30 mph")

### Autonomous mode


```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy


### Recources
* Self-Driving-Car Simulator: [Github Repo](https://github.com/udacity/self-driving-car-sim)
* Nvidia: [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
* Project specification: [Udacity Rubrics](https://review.udacity.com/#!/rubrics/432/view)
* Udacity repository: [CarND-Behavioral-Cloning-P3](https://github.com/udacity/CarND-Behavioral-Cloning-P3)
* [Udacity Self Driving Nanodegree](http://www.udacity.com/drive)