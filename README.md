# AN2DL-Challenge
In these challenges/projects we have been asked to perform **image classification** (challenge 1) and **data forecasting** (challenge 2) using Neural Networks and Deep Learning.

## Remarks about challenge 1
After two weeks of development we have been able to achieve an accuracy, on a hidden dataset, of **94.15%**.

### The task
In this homework you are required to classify images of leaves, 
which are divided into categories according to the species of the plant to which they belong. 
Being a classification problem, given an image, the goal is to predict the correct class label.

### The data
- Image size: 256x256
- Color space: RGB (read as 'rgb' in ImageDataGenerator.flow_from_directory ('color_mode' attribute) or use PIL.Image.open('imgname.jpg').convert('RGB'))
- File Format: JPG
- Number of classes: 14
- Classes:
  0. "Apple"
  1. "Blueberry"
  2. "Cherry"
  3. "Corn"
  4. "Grape"
  5. "Orange"
  6. "Peach"
  7. "Pepper"
  8. "Potato"
  9. "Raspberry"
  10. "Soybean"
  11. "Squash"
  12. "Strawberry"
  13. "Tomato"
- images per class:
  - Apple : 988
  - Blueberry : 467
  - Cherry : 583
  - Corn : 1206
  - Grape : 1458
  - Orange : 1748
  - Peach : 977
  - Pepper : 765
  - Potato : 716
  - Raspberry : 264
  - Soybean : 1616
  - Squash : 574
  - Strawberry : 673
  - Tomato : 5693

## Remarks about challenge 2
Until now, we have been able to achieve a RMSE (Root Mean Squared Error) of **3.7981** overall on a hidden test set.

### The task
In this homework, you are required to predict future samples of a multivariate time series. 
The goal is to design and implement forecasting models to learn how to exploit past observations in the input sequence to correctly predict the future. 

### The data
- Multivariate time series with the following characteristics:
  - Length of the time series (number of samples in the training set):   68528
  - Number of features: 7
  - Name of the features: 
    - 'Sponginess'
    - 'Wonder level'
    - 'Crunchiness'
    - 'Loudness on impact'
    - 'Meme creativity'
    - 'Soap slipperiness'
    - 'Hype root'
  - Uniform sampling rate.
