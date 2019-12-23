# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/random_samples_unique_class.png "Random Samples"
[image2]: ./writeup_images/distribution_classes_dataset.png "Dataset Distribution Chart"
[image3]: ./writeup_images/augmented_image_sample.png "Augmented Image Sample"
[image4]: ./writeup_images/augmented_dataset_distribution.png "Augmented Dataset Distribution Chart"
[image5]: ./writeup_images/clahe_sample.png "CLAHE Contrast Enhancement Example"
[image6]: ./writeup_images/modified_lenet_architecture.png "Modified LeNet Architecture"
[image7]: ./writeup_images/accuracy_with_and_without_dropout.png "Accuracy with and without Dropout"
[image8]: ./writeup_images/precision_recall_test_set.png "Precision and Recall of Test Set"
[image9]: ./writeup_images/web_test_images.png "Test Images from Web"
[image10]: ./writeup_images/prediction_web_test_images.png "Test Images from Web"
[image11]: ./writeup_images/softmax_prob_web_test_images.png "Softmax Probabilities of the web test images"
[image12]: ./writeup_images/visualization_conv1.png "Visualization of Convolutional Layer 1"
[image13]: ./writeup_images/visualization_conv2.png "Visualization of Convolutional Layer 2"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is the writeup. Here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32) as height and width with 3 channels
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The picture below shows one random sample belonging to each unique class label from the training set. It is observed that certain images are not clearly visible even to the human eye due to either low resolution, poor brightness or poor contrast.

![alt text][image1]

The distribution of each of the classes in the data set is also found using the pandas library and plotted below as a horizontal bar chart. The below chart shows that the dataset is biased and not equally distributed among all class labels. Some of the classes such as 'Go straight or left', 'Pedestrians', etc. have just around 200 images, while the other classes such as 'Yield', 'Priority Road', 'Speed limit (50km/h)', etc. have close to 2000 images. This shows that the data set is very imbalanced, which may lead to lower precision and recall for these classes. This means that augmentation of the dataset may be required to solve such issues.

![alt text][image2]



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The following preprocessing steps were performed on the data before training:

1. Augmentation of dataset for training

As mentioned before, it was observed that the dataset was imbalanced and additional images for certain labels could improve the training. As a result, every class label that had less than 1000 training images was augmented till it reached 1000 images. But it must be noted that there is a small risk of another bias creeping up - the leaking of the hints/results within the dataset. Since only certain signs are augmented and not all, only these signs may have the leftover patterns of transformation, rotation, etc. such as the black regions on the image borders.

The following operations were performed on the original training dataset using random parameters to reach the required 1000 images for each class label:

    a. Rotation - from -20deg to +20deg
    b. Translation - to both left and right upto 5 pixels
    c. Blurring - with cv2.blur function with kernel size from 2 to 5
    d. Noise addition - with uniformly distributed noise between -5 to 5 in RGB values 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the distribution of the original data set and the augmented data set is shown in the below bar chart.

![alt text][image4]

From the above chart, it can be seen that there is a minimum of 1000 images now for each class label and some of the existing bias in the dataset is reduced.

2. Contrast enhancement of images using CLAHE

As can be seen in the sample of images shown during exploration of the dataset, many of the images are in poor lighting conditions, with either low brightness or low contrast, which makes it difficult even for the human eye to recognize the signs properly. Inorder to solve this issue, all the training images, validation images and test images are processed to enhance the contrast using the OpenCV CLAHE function, which takes care of contrast enhancement not only at the global level, but also at local level in smaller regions of the images.

Here is an example of a poorly lit image on the left side and a contrast enhanced image using CLAHE on the right side.

![alt text][image5]

3. Normalization of the images

The channel values of all the training, validation and test images are normalized in the 0-1 range to enable better training.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				    |
| Fully connected		| 800x120, outputs 120  						|
| RELU  				|           									|
| Dropout  				| 0.5 probability   							|
| Fully connected		| 120x84, outputs 84    						|
| RELU  				|           									|
| Dropout  				| 0.5 probability								|
| Fully connected		| 84x10, outputs 43       						|

 
A diagram of this architecture is shown below:

![alt text][image6]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The training of the model was carried out using the Adam Optimizer to minimize the softmax cross-entropy loss function. This training was done in batches of 128 images for 30 epochs. A learning rate parameter of value 0.001 was used for the training.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were (as in the latest instance run on the [Jupyter notebook](./Traffic_Sign_Classifier.ipynb)):
* training set accuracy of 0.953
* validation set accuracy of 0.976
* test set accuracy of 0.966

The following approach was chosen to find the solution:
1. The existing LeNet architecture (used in the MNIST classification example) was trained initially on the dataset as it is.
2. The hyperparameters such as epochs, learning rate and batch size were varied one by one to find the best possible solution.
3. The dataset was augmented to reduce the imbalance. New images produced by translation, rotation, blurring and adding noise were created for the class labels with less than 1000 images.
4. Since the above steps could not yield results with greater than 94 percent validation accuracy, I decided to modify the architecture. The width of the network was tweaked. 16 layers in the first Conv2D layer and 32 layers in the second Conv2D layer were used instead of 6 and 16 layers used earlier.
5. To reduce overfitting, both batch normalization and dropout were tried out. Dropout layers were added after each of the first 2 fully connected layers and it was observed that they improved the performace. Batch normalization after each layer was not as effective as using dropout. The below plots show how the dropout layers helped in reducing the overfitting(a validation accuracy lower than the training accuracy indicates overfitting)
![alt text][image7] 
6. CLAHE contrast enhancement was done before the normalization of the images. This helped reach a validation accuracy close to 98 percent in many runs.

#### Precision and Recall for the Test Set#### 
The precision and recall for each of the class labels in the test set was evaluated to get an idea of how good the model performed on the different classes. It is observed that while most signs have high precision and recall rates around 0.9, certain signs such as 'Pedestrians', 'Double Curve', etc. have lower precision and recall rates around 0.6-0.7. There are also some signs such as 'End of speed and all passing limits' which have high recall close to 1, but lower precision around 0.8, which shows there could be significant number of images getting falsely classified as 'End of speed and all passing limits' (false positives). There is also the other case observed where there are many false negatives, such as 'Pedestrians' which has a relatively higher precision close to 0.8, but a lower recall of 0.6. It means some of the actual 'Pedestrians' signs are misclassified as some other sign. The below chart shows the precision and recall for each of the class labels in the test set:
 
![alt text][image8] 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image9] 

The first and the second image with 'Road Work' and 'Slippery Road' signs were chosen since they are clearer than most of the images in the dataset. I wanted to see if the classifier performs well on images that have more detail and clarity than most of the training images. Also, 'Slippery Road' had fewer images in the original dataset and therefore had to be augmented. It also makes sense to see if the augmented images were successful in training.

The third and the fourth image on 'Stop sign' and '120kmph Speed Limit' might be difficult to classify because they are in a different perspective and not perpendicular. I wanted to check if images from a slightly different perspective could still be classified. The 'Stop sign' also did not have many images in the dataset and had to be trained with some augmented images. The '120kmph Speed Limit' might also get misclassified due to similar looking signs like 20kmph or 70kmph speed limit. These images were selected to check if any of these issues occur.

The fifth image was chosen such that the actual sign is not exactly cropped, but has an extra board at the bottom with letters, due to which it is slightly offset towards top from the centre. Also, there is a sign with a similar distribution of pixels, the 'Traffic lights' sign, into which it might get misclassified. (if colour inside the sign is not considered by the network for classification, then both the signs would look similar, especially with low resolution images)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image10] 

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work    									| 
| Slippery Road     	| Slippery Road 								|
| Stop					| Stop											|
| Speed limit (120 km/h)| Speed limit (20 km/h)							|
| General caution		| Bicycles crossing    							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This shows that the training needs to be improved to reduce these errors. One of the errors, the '120kmph' getting classified as '20kmph', was expected while choosing the images, and is confirmed, while the second error of 'General caution' being classified as 'Bicycles crossing' is unexpected. Analysis of the softmax probabilites, as done in the next section below, would help us understand this better.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for finding the softmax predictions on the test images from the web is located in the second last code cell of the [Jupyter notebook.](./Traffic_Sign_Classifier.ipynb). 

The following horizontal bar charts show the top 5 calculated softmax probabilities for each of the 5 test images downloaded from the web:

![alt text][image11] 

The analysis of the softmax probabilities and the predictions is listed below for each of the images:

1. For the first image, the model is very sure that this is a 'Road work' sign (probability close to 1.0), and the image does contain a 'Road work' sign. 

2. For the second image, the model is almost sure that this is a 'Slippery Road' sign (probability close to 0.99). The models assigns a probability of around 0.01 to the 'Dangerous curve to the left' sign, maybe due to the distribution of the black pixels inside the red triangle being similar to that of the 'Slippery Road' sign.

3. For the third image, the model is very sure that this is a 'Stop' sign (probability close to 1.0)

4. For the fourth image, the model classifies it as '20kmph' sign with a high confidence (probability close to 0.98). This is wrong, since actually the image contains a '120kmph' sign. The model assigns a probability of 0.02 to the 70 kmph sign, where the digit '7' has a similar shape to the digit '2' in the '120kmph' sign. 

   Due to the perspective, the digit '1' in the sign is shorter and maybe not recognized distinctly from the red outer circle in the sign by the network layers. One consolation is that the '120kmph' sign does find a place in the top 5 probabilites, which means that the model is not totally unreasonable.

5. For the fifth and final image, the model again makes a mistake. It misclassifies the 'General caution' sign as 'Bicycle crossing', but not with a high confidence level (probability close to 0.55). This could be again due to slightly similar distribution of bright and dark pixels inside the red triangle and the low resolution of the bicycle images, where distinct features are not recognizable, even to the human eye. 

   Another interesting fact is that the 'Bicycle crossing' sign shows a lower precision and a higher recall in the given test dataset, indicating high possibility of false positives. Similarly, the 'General caution' sign shows a higher precision and lower recall in the test data, indicating high possibility of false negatives. This has been confirmed now with the web test images!

   The model lists the 'Traffic lights' sign as the second best match with a probability close to 0.38. This was expected even before the training, since the pixel arrangement is very similar to that of the 'General Caution' sign. This indicates that the colour of the pixels in the images, probably atleast inside the sign borders is not given a high weightage in the training. The actual sign in the image, 'General caution', is listed as the third best match with a probability around 0.03. 

The above results show us that the model is not totally foolproof, and that better data and training are required. But, at the same time, we have found the model to be not so unreasonable and found it working good in most cases. 

To conclude, we can say that this approach works, but needs to be improved with better data, a better network architecture and better hyperparameters.



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The code related to visualization of the neural network can be found in the last code cell of the [Jupyter notebook.](./Traffic_Sign_Classifier.ipynb)

The 4th image in the web test set has been fed in as a sample input to the visualization code. The visualization of the output of the first convolution layer is shown below:

![alt text][image12] 

It can be seen that the network successfully segregates the important features such as the outer circle of the sign board, and the digits inside. Some of the feature maps don't show 120 clearly, with some not showing the digit '1' clearly (feature map 7), some not showing the digit '2' clearly (feature map 11), but most of the feature maps show all the 3 digits clearly.

The visualization of the output of the second convolution layer is shown here:
![alt text][image13] 

From this picture, I could not clearly understand the activations, but I think one can definitely see patterns of the outer circular border of the sign, and also some diagonal strikes in feature maps 11 and 29 (these diagonal patterns may not be activated much with this input sign, but probably may be highly activated with those signs that have such diagonal strikes or arrows).

The visualizations of the layers and maybe even weights can help us understand in detail what each layer of the network does. It can be an important tool to understand and thereby improve the performance of the network.
