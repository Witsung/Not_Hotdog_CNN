# Not_Hotdog_CNN
An image classifier that applies convolutional neural network to tell the users what is not a hotdog.

This creation of this model is inspired by the tv series Silicon Valley, which Jian yang demonstrates his very useful and successful "Not Hotdog App".
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/ACmydtFDTGs/0.jpg)](https://www.youtube.com/watch?v=ACmydtFDTGs)

I identified the drawbacks of this image classifier and try to make improvement base on Jian yang's idea. Thus, I decided to extend the functions of the app to make it possible to identify pizza as well! This function turns the classifier into a Hotdog/Pizza images classifier.

## Data Collection
The training data for the model is acquired from ImageNet, a database that is designed for computer vision research. For collecting hotdog images, "hotdog", "sausage", "frankfurter" are used. For pizza, the keywords "pizza" is already enough to collect comparable amounts as the hotdog.

## Data Preparation/Augmentation
Since we have very little amount of images, about 1500 for each category. We can utilize the augmentation methods that are already provided by Keras to fully utilize the images and to prevent the model from overfitting as well. For instance, zom_range can randomly zoom in images. These methods can prevent the model from training with the exactly same image again. The augmentation techniques I used are as follow.
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
The model is constructed by using Keras' sequential model.

## Model Evaluation
