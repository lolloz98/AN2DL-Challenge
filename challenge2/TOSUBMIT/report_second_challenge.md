# Challenge 2
For this challenge we had a lot of ideas, explored many of them and left out others, due to time constraints. 

We have mainly focus on trying to understand how to create a good model which could give good predictions, how to set the parameters for the model and how to handle the dataset in a way to maximize the learning curve of the model.
In particular, for this last point, we have tried many different options (we will present only a few of them here).

We have not tried other loss functions other then mse and we have not tried to use other optimizers (we used only Adam which, on the paper, seemed to be the best). We tried a couple of times to use different activation fucntions and different learning rates (also using customized scheduling), but we did not go in depth with these ideas. Also, we have almost always used minmax normalization, even though sometimes we have also used standard normalization (or no normalization at all: why not try?).

We have always tried to use a compact notebook (similar to the one seen in class), to easily run and train different models using just one click. In our `final` notebook we have placed uself functions to easily switch normalization method (standard or minmax) and we have changed a little bit the way of dividing the dataset into chuncks: we have preferred to remove the first samples rather then to use them with padding. We have added a function to shuffle our sequences, so to be able to have different validation sequences for the training. At last, we have decided to remove completely the test set because we have thought that using validation set would be good enough to our evaluation in the local environment (indeed, we wanted to have only a really rough estimate on how the model was performing -> only to roughly understand if it was worth to submit or not. We had plenty of submissions to make this kind of reasoning).

## Baseline models
As baseline model we have tried to implement a simple FFNN. However, for larger telescope size, as excpected, the performances were rather poor (very noisy output) and we decided not to submit the model.

So we have decided to use the model seen in class as baseline. 
Even though, to be able to actually predict something and not just a mean we had to try few different parameters, after handling correctly the normalization and denormalization in the model file, we were able to obtain straight away the quite good result of `4.1996984482`.

We have then tried to tweak this model using only LSTM rather then Bidirectional LSTM (just to make some trials and understand a bit better how the whole thing worked). We have also tried to stack multiple LSTM layers and to use a small dense fully connected ouptut to the network.
This trials have not led to better results in the local environment and we have only published on codalab few of them.
However, this whole process was useful, both to do some trial and error, and to understand a bit better the 'right' dimensions of the `window`, `telescope` and `stride` params.
Indeed, for our next models and next hypothesis we have kept them almost always (sometimes we have tried different params, such as `telescope=288` or different initialization) equal to `window=600`, `telescope=108` and `stride=10`.

After having a look at the data (we have computed the correlation between the various signals and actually look at the data) we have seen that there were some hint on what could be done.
First of all, some sequences of data were rather strange: there were long sequences of ones which, to us, seemed like the data was corrupted; moreover, only a couple of sequences had high correlation (`Crunchiness` and `Hype root`) whereas, the other had particulalrly low correlation.

## Multimodel approach
To exploit the correlation factor and because many of the models gave as output the mean of the input signal, we thought best to simplify the input to the model and to divide the problem into more separate models. One model should compute: `Crunchiness` and `Hyperoot` given the two sequences, another one `Wonder level` and `Loudness on impact` and so on.

This, obviously, increased the training time by a lot (we wanted to train 5 models instead of 1!) but we thought it could give nice results. However, we were proved wrong: the results were quite disappointing.

## Handling dataset 'noise' and 'corruption'
As stated above, the long series of ones in the signals, which were present here and there in the dataset, make us suspicious of dataset corruption. Moreover, we really thought that it would have been better to smooth the data altogether; we thought: can high frequency feature be really learned? (however, at last we re-thought that it could have been actually a mistake, to try to smooth out the dataset in the way we did. We have always smoothed the 7 signals separatly from each other -> this could have generated some loss of information which could have turned out in worse prediction capabilies).

To handle the corrupted data and/or smooth the data we have tried really a lot of different approaches.
- remove the ones series by averaging the 'closer' points not set to one
- using regression on each signal to smooth it out through low polynomials (we had to split each sequence in many sub-sequences and then to reconstruct the sequence: look at `data_smoothing/smooth_with_regression_4.ipynb`)
- using fft and ifft (we wanted to remove only high frequencies) (this was kind of useless thing to try out: the theory was against us, but we were curious)
- using butter lowpass filter to filter out high-freq (this made the signal shift on the right... not really good smoothed dataset) (look at `data_smoothing/smooth_with_lowpass.ipynb`)
- using autoencoder (for this approach we used a dense network: it was not the best idea, since the output was noisier then the input. We should have used an autoencoder with lstm. However, we did not try because we were already quite satisfied with the results of the regression approach) (look at `data_smoothing/smooth_with_autoencoder.ipynb`)

At last we have decided (also considering the mse between the newly computed dataset and old one) that the regression dataset was the best (we tried different values for stride, size -> look at notebook)

## Model
By searching online and trying out different option, the model that we have found to work best, both for training time and for the accuracy of the predictions have been `e1d1`. 

This model works with an encoder-decoder architecture and it seems really effective to solve our problems. With a dataset smoothed with regression we have been able to score the very good result of `3.7424969673`.

We have tried then many variation on the theme: we have tried to use convolution before the lstm, we have tried to use bidirectional lstm (the time for trainig went up by far), we have tried to use skip connections and stacking different lstm blocks (modeled as encoder-decoder) over that architecture, without obtaining better resuts.

We have also noticed that on codalab the most important prediction to make to get a better overall rmse were on `Crunchiness` and `Hype root`. We have worked hard to improve these two sequences (also retrying the multimodel approach) by using `weights` for the input sequences (we have tried to multiply the normalized input sequences by the rmse output in codalab. We thought that with this we could have achieved different weighting for each sequences, thus we would have predicted better the highest weighted sequences, but actually we had no improvement...) and other techniques.

In the last couple of days of the challenge, we have dropped process of dataset smoothing and we have made a couple of very promising models (in local they are by far the best one we had done) using the original training dataset and simply discarding the sequences which we thought to be corrupted (look at the `chunk` fucntion in `final_notebook.ipynb`). However, due to codalab malfunctions we have not been able to run it on the hidden test set.

## Final considerations
This challenge was particularly intriguing from our part because we have really had much trouble to improve the predictions: we have tried many things but nothing really seemed to really affect the prediction power of the network. We had to rethink all of our step and to go back to revisit our assumptions and change them.

At last we have come up with a good solution and we are quite happy (we think: we have not been able to test it on codalab).