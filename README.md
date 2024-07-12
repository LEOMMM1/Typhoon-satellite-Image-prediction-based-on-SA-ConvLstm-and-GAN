### Overview
Predict the typhoon satellite image based on Self-Attention Convlstm and GAN neural Networks.
I refer to these paper mainly:
[Self-Attention ConvLSTM for Spatiotemporal Prediction](https://doi.org/10.1609/aaai.v34i07.6819)
[Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://paperswithcode.com/paper/convolutional-lstm-network-a-machine-learning)
[Satellite Image Prediction Relying on GAN and LSTM Neural Networks](https://ieeexplore.ieee.org/abstract/document/8761462)

## Dataset introduction
The satellite image dataset is from Himawari-8,including three channels:Band 8,Band 9, Band 10.
I use the Band 10 as the input data.
Training dataset and validation dataset include four types:A,B,C,E and the test dataset are five types:U,V,W,X,Y.The image was token every hour in every type.

## Loss function 
The loss function is composed of three part:ssim(structural similarity), reconstruction loss(L1 loss+L2 loss), adversarial loss.

## Metrics
Metrics include mse,ssim,psnr(peak signal-to-noise ratio),sharpness.The method of calculating sharpness is from [No-Reference Image Sharpness Assessment Based on Maximum Gradient and Variability of Gradients](https://ieeexplore.ieee.org/document/8168377)

