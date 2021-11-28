# Challenge1 Report

In this competition we have mainly tried to use pretrained model to try to maximize accuracy on our dataset by using transfer learning and fine tuning.

The parameters on which we have mostly focused on have been:
1. How good a certain model was for our task
    - we have tried a lot of different models: ResNet, DenseNet, EfficientNet, InceptioNet, NasNet, VGG16...
    - we have tried a lot of variations of the cited models.
2. How changes in hyper-parameters or in the classifier could affect the learning of our models
    - we changed patience for EarlyStopping: mostly we kept it to 5, but we have done some trials with 10.
    - we changed classifier: from fully connected hidden layers to a smaller fully connected hidden layer enriched with GAP and Batch Normalization layers. We have also made changes in the dropout configurations and other small changes.
    - we tried to fine tune with different parameters.
3. change in the structure of the dataset.
4. Ensemble various models to get more precise results without the need of time to train a new model.

We have also tried to create our own NN following the guidelines of ResNet and EfficientNet, but trying to make it smaller for avoiding both overfitting and expensive training.

We were not able to explore the usage of different loss functions nor optimizers among other things. We have always used Adam optimizer and CrossEntropy as a loss function.

Even though we have read that, sometimes, SGD or other optimizers (for example RMSProp which, from what we understand, has been used for the training of EfficientNet) can achieve smoother minimums, in this challenge we have decided to focus on other aspects and hyper-parameters rather than these ones.

We have evaluated performance mainly via: confusion matrix and accuracy.

We have also computed other performance metrics such as F1, recall and precision. However we have found that given the disproportion in the dataset, the confusion matrix was the most useful tool to evaluate performance.
### Transfer Learning
From the point of view of transfer learning we have tried many different solutions.

Since it was our first project on Neural Network we were rather curious about the possibility of different neural networks. So, by following the networks' ratings on the ImageNet dataset we tried many of the most promising models in terms of accuracy and in terms of FLOPS.

At first we have implemented VGG16 as a baseline, because we wanted to make sure that our workflow was correct and because we had already implemented it in class.

We have tried to create a pipeline as general as possible. Thanks to this, which you can find in the notebooks, we were able to modify the model by choosing different hyper-parameters or different 'supernets' by modifying only a couple of lines of code.

The first models seemed really promising in our local environment. But on the hidden test we could not score more than about 0.6.

We were disappointed, because on our validation set we achieved results over 0.9 with the same models.

So after a couple of trials with different nets, which all reported that the results on the hidden test-set were much different from the results obtained by our validation set and by looking at partial accuracy-score of our submissions we understood that the actual problem was not the starting network but the dataset.

We have tried firstly to take a balanced random subset of the dataset: we thought that the main problem could be data scarcity and the disproportion of `tomatos` w.r.t. the other leaves. With this approach we have not been able to really get results. We thought it was a problem of underfitting.

Then, we have tried to use class_weights. We have computed them by simply looking at the number of images for each class (the higher the number of images, the lower the weight) and by normalizing the result. With ResNet50 we were suddenly able to hit the 0.75 score.

We have also done a trial of preprocessing images by automatically obscuring the background. However, this submission was discarded because opencv-python was not supported by the remote environment. We could have used Pillow instead, but meanwhile we have found a better way to improve our accuracy on the hidden test-set, so we have dropped the idea.

By paying more attention to the data given to us, we have been able to spot that in almost each class there were major differences between the healthy and unhealthy leaves. After careful examination we have manually split the dataset and trained our models on a larger output space.

With many models: EfficientNet, Inception_ResNet_v_2, DenseNet... We have been able to improve our score on the hidden test-set up to 0.89.

In the meanwhile we have also discovered that a smaller classifier worked best instead of a big classifier with a hidden layer. Our final decision has been to train the network using a GAP layer (followed by a small FC layer). We found that this worked particularly well in general. Our explanation and intuition was based on the shift invariance property of this layer.

Since many of our colleagues were able to go higher than 0.9 we felt the need to compete once again and we have done a better splitting of the dataset. By adding more classes, with a more accurate manual division, the dataset we were able to achieve a score of 0.913 (trained with an EfficientNetB1 with a fine tuning of the last block).

For the disproportion in the data provided among the classes we have decided to keep the weights (computed automatically by a handy function in our pipeline). We have also made our notebook totally independent from the number of classes of the dataset: in this way we were able to switch datasets comfortably by simply setting a couple of parameters.

After trying many other models (MobileNet, other ResNets, EfficientNets of different dimensions) we have decided to try to ensemble some of our best nets together and to see if we would be able to bump up our score on the hidden set.

By using two models we were indeed able to achieve an accuracy of 0.945 on the hidden test-set. We have tried, for simplicity, a max ensemble methods (we returned the index which we were 'most certain of') and a sum-ensemble methods (please see the related submission's `model.py`) (for specific settings look at `best efficient model.ipynb` and `best inception model.ipynb` in `some_of_our_runs/more_division` folder).

We have also found online the original dataset with backgrounds and for curiosity we have also trained a couple of models on this dataset. We were able to score 0.91 (with an ensemble of two nets, one trained on this new dataset and the other on the previous dataset).
### Our Net
After we have been able to achieve the 0.9 score result, we have decided to try to design our own NN.

The main requirements that we wanted it to follow were: small training time, small overall in space.

Thus, we have tried to make a model with few layers.
After reading the paper of ResNet, we were quite curious to see the performance of the net if we used skip connections. In this net we did not think that they were so necessary since the model was rather small.

We tried to make the model grow more in width than in depth (we though that maybe we could achieve better performance in this way, even though we were not sure of this idea) and we have trained the model on the latest dataset.

Unfortunately our training time was rather long. We blamed the initialization and the width of the layers (last layer had 256 filters).

We were able to obtain on the test-set a score of 0.68.

We have also tried to simplify the net but without much benefit in terms of accuracy (accuracy dropped on our validation set and we decided not to submit this model).

Overall we were satisfied with this model. We did not expect great results because of the initializations and the rather small dataset. It would have been best to train it on ImageNet and then to fine tune it on our dataset.

### Some considerations
The flow of ideas during this challenge seemed to never stop. Due to the long training time and the lack of hardware resources (only one of us was able to run in a feasible amount of time a net on his local machine, and both kaggle and colab gave limited resources) we had to take major decisions on what to try and what not to try.

We have given more importance to try many and different pretrained models by using fine-tuning. We thought it to be very interesting, because thanks to this selection we were able to understand a bit better the application of transfer learning, its possibilities and its limits.

Another thing which we found fascinating has been discovering that many times, small models such as DenseNet or EfficientNetB0 were able to obtain better results than much larger models such as EfficientNetB7, ResNet50 or NasNetLarge (probably because of some overfitting issues).

It could have been a good idea to train the final network on the whole dataset (by giving a precise number of epochs in training). However, we were already satisfied with the results that we have got so we did not try this approach.
