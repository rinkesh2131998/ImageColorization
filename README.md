# Version 1(CNN AutoEncode-Decoder)

## ImageColorization
- Implemented the model based on the paper [Colorful Image Colorization](https://arxiv.org/abs/1603.08511)
- Implemented in Keras
- Custom Dataset built by taking an youtube video of mario gameplay and extacting frames.

## Training
- Run the `train_script.py` for traing any custom dataset
- Run `colorization.py` to convert images from grayscale to rgb

## CNN Model Architecture:
<img src=https://github.com/rinkesh2131998/ImageColorization/blob/master/models/encoder.jpeg" width="300" height="800"/>

<br />

# Version 2 (PIX2PIX GAN)

- This is the updated Image Colorization model built using the PIX2PIX Gan architecture
- I used a different dataset to train this network, which was collected from Pixabay
- Using Gan for Image colorization provided better results as compared to `Version 1` of the model
- The input images for the network should be 256*256 in height and width for the training

- `./Gans/trainGan`: Run this script for training the model on the dataset
- `./Gans/imgColor`: Use this script to convert <strong>GrayScale to RGB</strong> images

## Generator Model:
<img src="https://github.com/rinkesh2131998/ImageColorization/blob/master/models/GANmodelImages/genrator.jpg" width="300" height="800"/>

## Discriminator Model:
<img src="https://github.com/rinkesh2131998/ImageColorization/blob/master/models/GANmodelImages/discriminator.jpg" width="300" height="800"/>

### Converted Image Samples:

#### 1.Ground Truth&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.GrayScale Version &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.Gan Output
![samples](https://github.com/rinkesh2131998/ImageColorization/blob/master/OutputImages/ganOutputs/9921af24-6167-46ed-9fe2-de6d7bac240d.png)
