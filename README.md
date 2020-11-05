# Version 1(CNN AutoEncode-Decoder)

## ImageColorization
- Implemented the model based on the paper [Colorful Image Colorization](https://arxiv.org/abs/1603.08511)
- Implemented in Keras
- Custom Dataset built by taking an youtube video of mario gameplay and extacting frames

## Training
- Run the `train_script.py` for training any custom dataset
- Run `colorization.py` to convert images from grayscale to rgb

## CNN Model Architecture:
<img src="https://github.com/rinkesh2131998/ImageColorization/blob/master/models/encoder.jpeg" width="300" height="800"/>

### Comparision of output of both the models

#### 1.Ground Truth&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.Cnn Output(Version 1)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.GAN Output(Version 2)
- <img src ="https://github.com/rinkesh2131998/ImageColorization/blob/master/OutputImages/inputImages/image574.jpg" width="175" />
  <img src ="https://github.com/rinkesh2131998/ImageColorization/blob/master/OutputImages/output2final/test5.png" width="175" />
  <img src ="https://github.com/rinkesh2131998/ImageColorization/blob/master/OutputImages/ganOutputs/img1.jfif" width="175" />
- <img src ="https://github.com/rinkesh2131998/ImageColorization/blob/master/OutputImages/inputImages/image933.jpg" width="175" />
  <img src ="https://github.com/rinkesh2131998/ImageColorization/blob/master/OutputImages/output2final/test9.png" width="175" />
  <img src ="https://github.com/rinkesh2131998/ImageColorization/blob/master/OutputImages/ganOutputs/img2.jfif" width="175" />
- <img src ="https://github.com/rinkesh2131998/ImageColorization/blob/master/OutputImages/inputImages/frame1313.jpg" width="175" />
  <img src ="https://github.com/rinkesh2131998/ImageColorization/blob/master/OutputImages/output2final/test14.png" width="175" />
  <img src ="https://github.com/rinkesh2131998/ImageColorization/blob/master/OutputImages/ganOutputs/img0.jfif" width="175" />



<br />

# Version 2 (PIX2PIX GAN)

- This is the updated Image Colorization model built using the PIX2PIX Gan architecture
- Due to great results on the first dataset, I used a different dataset, collected from Pixabay and containing more vibrant colors to train this network
- Using Gan for Image colorization provided better results as compared to `Version 1` of the model
- The input images for the network should be 256*256 in height and width for the training

- `./Gans/trainGan`: Run this script for training the model on the dataset
- `./Gans/imgColor`: Use this script to convert <strong>GrayScale to RGB</strong> images

## Generator Model:
<img src="https://github.com/rinkesh2131998/ImageColorization/blob/master/models/GANmodelImages/genrator.jpg" width="300" height="800"/>

## Discriminator Model:
<img src="https://github.com/rinkesh2131998/ImageColorization/blob/master/models/GANmodelImages/discriminator.jpg" width="300" height="800"/>

### Version 2 Converted Image Samples:

#### 1.Ground Truth&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.GrayScale Version &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.Gan Output
![samples](https://github.com/rinkesh2131998/ImageColorization/blob/master/OutputImages/ganOutputs/9921af24-6167-46ed-9fe2-de6d7bac240d.png)

