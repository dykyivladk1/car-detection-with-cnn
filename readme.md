# Vehicle Detection and Localization

This repository contains a PyTorch implementation of a vehicle detection and localization model. The codebase is designed to train a Convolutional Neural Network (CNN) to identify the bounding boxes of vehicles in images.

## Overview

The implementation reads vehicle image data and corresponding bounding box coordinates, normalizes the bounding box data, applies image transformations, splits the data into training and testing sets, defines a CNN architecture for regression, trains the model, and evaluates its performance on the test set.

## Features

- Data normalization using MinMaxScaler for bounding box coordinates.
- Custom PyTorch Dataset class to handle image loading and transformation.
- Convolutional Neural Network definition with PyTorch for bounding box prediction.
- Training and evaluation loops with model checkpointing.
- Visualization of prediction results with actual and predicted bounding boxes on the test images.

## Dependencies

- PyTorch
- Pandas
- Scikit-learn
- OpenCV
- Pillow
- Matplotlib

## Usage

Set the `root` variable to the path containing your training images and the corresponding CSV file with bounding box coordinates. The CSV file should have the columns `xmin`, `ymin`, `xmax`, `ymax`, and `image`.

To train the model, uncomment the training loop and run the script. Once the model is trained, it will be saved to `model.pth`. To evaluate the model, load `model.pth`, and run the inference code provided at the end of the script.

## Model Architecture

The CNN consists of five convolutional layers with ReLU activations and max-pooling, followed by four fully connected layers. The final layer outputs four values corresponding to the normalized coordinates of the vehicle's bounding box.

## Note

The current codebase is set up for inference only, with the training loop commented out. To train the model, ensure that you uncomment the training loop and have the necessary data in the specified `root` directory.
