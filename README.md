# Version 1(CNN AutoEncode-Decoder)

## ImageColorization
- Implemented the model based on the paper [Colorful Image Colorization](https://arxiv.org/abs/1603.08511)
- Implemented in Keras
- Custom Dataset built by taking an youtube video of mario gameplay and extacting frames.

## Training
- Run the `train_script.py` for traing any custom dataset
- Run `colorization.py` to convert images from grayscale to rgb

<br>

# Version 2 (PIX2PIX GAN)

- This is the updated Image Colorization model built using the PIX2PIX Gan architecture
- I used a different dataset to train this network, which was collected from Pixabay
- Using Gan for Image colorization provided better results as compared to `Version 1` of the model
- The input images for the network should be 256*256 in height and width for the training

`./Gans/trainGan`: Run this script for training the model on the dataset
`./Gans/imgColor`: Use this script to convert <b>GrayScale to RGB</b> images

## Generator Model:
![Genrator]()

## Discriminator Model:
![Discriminator]()

### Converted Image Samples:

#### Original Image /t GrayScale Image /t Gan outPut Image
![samples]()
