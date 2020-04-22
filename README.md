# Image-Classifier-CNN
Convolutional Neural Network image classifier implemented in Keras, for distinguishing between Cats and Dogs
Performance
Dataset: Dogs vs Cats

Description: Binary classification. Two classes two distinguish - dogs and cats.

Training: 6400 images in total for both classes (3200 for each class)

Validation: 1600, 800 per each class


Model 1
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_13_input (InputLayer) (None, 320, 320, 3)       0         
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 320, 320, 32)      896       
_________________________________________________________________
max_pooling2d_12 (MaxPooling (None, 160, 160, 32)      0         
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 160, 160, 64)      18496     
_________________________________________________________________
max_pooling2d_13 (MaxPooling (None, 80, 80, 64)        0         
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 80, 80, 128)       73856     
_________________________________________________________________
max_pooling2d_14 (MaxPooling (None, 40, 40, 128)       0         
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 40, 40, 32)        36896     
_________________________________________________________________
max_pooling2d_15 (MaxPooling (None, 20, 20, 32)        0         
_________________________________________________________________
conv2d_17 (Conv2D)           (None, 20, 20, 64)        18496     
_________________________________________________________________
max_pooling2d_16 (MaxPooling (None, 10, 10, 64)        0         
_________________________________________________________________
dense_7 (Dense)              (None, 10, 10, 128)       8320      
_________________________________________________________________
global_average_pooling2d_3 ( (None, 128)               0         
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 129       
=================================================================
Total params: 157,089
Trainable params: 157,089
Non-trainable params: 0
_________________________________________________________________


Processing
CPU - i7 8550U 
Ram - 8Gb

