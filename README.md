# Not_Hotdog_CNN
An image classifier that applies convolutional neural network to tell the users what is not a hotdog.

This creation of this model is inspired by the tv series Silicon Valley, which Jian yang demonstrates his very useful and successful "Not Hotdog App".

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/ACmydtFDTGs/0.jpg)](https://www.youtube.com/watch?v=ACmydtFDTGs)

I identified the drawbacks of this image classifier and try to make improvement base on Jian yang's idea. Thus, I decided to extend the functions of the app to make it possible to identify pizza as well! This function turns the classifier into a Hotdog/Pizza images classifier.

## Data Collection
The training data for the model is acquired from ImageNet, a database that is designed for computer vision research. For collecting hotdog images, "hotdog", "sausage", "frankfurter" are used. For pizza, the keywords "pizza" is already enough to collect comparable amounts as the hotdog.

## Data Preparation/Augmentation
Since we have very little amount of images, about 1500 each category. We can utilize the augmentation methods that are already provided by Keras to fully utilize the images and to prevent the model from overfitting. For instance, ```zom_range``` can randomly zoom in images. These methods can prevent the model from training with the same image again. The augmentation techniques I used are as follow.
```python
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    height_shift_range = 0.3,
    width_shift_range=0.2,
    rotation_range=3,
    zoom_range=[0.2, 0.6],
    horizontal_flip=True,
    vertical_flip=True
)
```

## Building CNN Model Using Keras
The model is constructed by using Keras' sequential model. The structure of the model can be seen as below. Every cov layer is attached with 'relu' activation function, as it is said to be one of the best practice to start with. Only the last fully connected layer uses 'sigmoid' to generate probability output in order to be transformed to binary classification. 

The network capacity (number of layers, number of filters of cov) is decided by try and error. If the validation loss is decreasing as the capacity growth, I keep increasing the capacity until the validation loss increased. In addition, pooling layers and dropout layers are added if overfitting can be observed. Learning rate is set to 0.002 by default in ```keras.optimizers.Adamax```, although I play around a bit with it, 0.002 seems to be the most promising one.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 149, 149, 32)      416       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 73, 73, 32)        4128      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 36, 36, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 35, 35, 32)        4128      
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 34, 34, 64)        8256      
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 33, 33, 128)       32896     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 11, 11, 128)       0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 15488)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               1982592   
_________________________________________________________________
activation_1 (Activation)    (None, 128)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 129       
=================================================================
Total params: 2,032,545
Trainable params: 2,032,545
Non-trainable params: 0
```
_________________________________________________________________

## Model Evaluation
![Validation Score](https://github.com/Witsung/Not_Hotdog_CNN/blob/master/Validation%20Score.png)

Early stopping is used to stop training the model when the validation score is not improving in 5 epochs. So the training stops at 20th epochs.

![Accuracy](https://github.com/Witsung/Not_Hotdog_CNN/blob/master/Accuracy.png)

As one can see from the graph, the training accuracy is generally below test accuracy and the training stops before the model starts to overfit. The accuracy of the final epoch is 82%.
