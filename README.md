# Facial-Keypoint-Detection

## Project Overview

  Here we're defining and training a convolutional neural network to perform facial keypoint detection.


Let's take a look at some examples of images and corresponding facial keypoints.

![facial points](https://bewagner.net/programming/2020/04/23/detecting-face-keypoints-with-opencv/)


Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in the image above. In each training and test image, there is a single face and **68 keypoints, with coordinates (x, y)**, for that face. These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. Here they are, numbered, and you can see that specific ranges of points match different portions of the face.

![68 keypoints](https://github.com/MariaSimon-AI/Facial-keypoint-Detection-using-Pytorch/blob/main/images/landmarks_numbered.jpg)

## Local Environment Instructions

Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.

```
git clone https://github.com/MariaSimon-AI/Facial-keypoint-Detection-using-Pytorch.git
cd P1_Facial_Keypoints

```
Create (and activate) a new environment, named cv-nd with Python 3.6. If prompted to proceed with the install (Proceed [y]/n) type y.

**Linux or Mac:**
```
conda create -n cv-nd python=3.6
source activate cv-nd
```
**Windows:**

```
conda create --name cv-nd python=3.6
activate cv-nd
```
At this point your command line should look something like:
```
 (cv-nd) <User>:P1_Facial_Keypoints <user>$.
```
 The (cv-nd) indicates that your environment has been activated, and you can proceed with further package installations.

Install PyTorch and torchvision; this should install the latest version of PyTorch.

**Linux or Mac:**

```
conda install pytorch torchvision -c pytorch
```
**Windows:**
```
conda install pytorch-cpu -c pytorch
pip install torchvision
```
Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```
## Data

We will be using the YouTube Faces Dataset. It is a dataset that contains 3,425 face videos designed for studying the problem of unconstrained face recognition in videos. These videos have been fed through processing steps and turned into sets of image frames containing one face and the associated keypoints.

**Training and Test Data**

This facial keypoints dataset consists of 5770 color images. All of these images are separated into either a training or a test set of data.

- 3462 of these images are training images, for you to use as you create a model to predict keypoints.
-2308 are test images, which will be used to test the accuracy of your model.

All of the data you'll need to train a neural network is in the Facial-Keypoint-Detection repo, in the subdirectory data. In this folder are training and tests set of image/keypoint data, and their respective csv files. This will be further explored in Notebook 1: Loading and Visualizing Data, and you're encouraged to look trough these folders on your own, too.

## Project Structure
The project is broken into a few main parts in four Python notebooks:

- models.py

- Notebook 1 : Loading and Visualizing the Facial Keypoint Data

- Notebook 2 : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

- Notebook 3 : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

- Notebook 4 : Fun Filters and Keypoint Uses


1. Navigate back to the repo. (Also, your source environment should still be activated at this point.)
```
cd
cd Facial-Keypoint-Detection
```
2. Open the directory of notebooks, using the below command. You'll see all of the project files appear in your local environment; open the first notebook and follow the instructions.
```
jupyter notebook
```
3. Once you open any of the project notebooks, make sure you are in the correct cv-nd environment by clicking Kernel > Change Kernel > cv-nd.
