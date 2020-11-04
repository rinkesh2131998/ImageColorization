import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb,rgb2lab
from skimage.io import imsave
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

##############################################################
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True, help='Path to folder with the generator save model in .h5')
parser.add_argument('-s', '--source', required=True, help='Directory with all images to convert to RGB')
parser.add_argument('-t', '--target', required=True, help='Target directory to store converted output images')
args = vars(parser.parse_args())

##############################################################
sourcePath = args['source']
destPath = args['target']

if not os.path.exists(sourcePath):
    os.makedirs(sourcePath)
if not os.path.exists(destPath):
    os.makedirs(destPath)

##############################################################
def lab_to_img(img):
  canvas = np.zeros((256,256,3))
  for i in range(len(img)):
    for j in range(len(img[i])):
      pix = img[i,j]
      canvas[i,j] = [(pix[0] + 1) * 50,(pix[1] +1) / 2 * 255 - 128,(pix[2] +1) / 2 * 255 - 128]
  print(np.max(canvas[2]), np.min(canvas))
  canvas = (lab2rgb(canvas)*255.0).astype('uint8')
  print(np.max(canvas), np.min(canvas))
  return canvas

##############################################################
X = []
for f in os.listdir(sourcePath+'/'):
  x = img_to_array(load_img(sourcePath+'/'+f))
  if x.shape[2] == 3:
      x = rgb2lab((1.0/255 * x))
      x = x[:,:,0]
  else:
      x = 1.0/255 * x
  X.append(x)
X = np.array(X, dtype='float32')
X = (X/50) - 1
X = X.reshape((X.shape[0], 256,256, 1))

##############################################################
model = tf.keras.models.load_model(args['path'])

##############################################################
pred = model.predict(X)
output = []
for i in range(len(X)):
  l = X[i]
  ab = pred[i]
  imgOut = lab_to_img(np.dstack((l, ab)))
  output.append(imgOut)

##############################################################
cnt = 0
for i in output:
    cnt+=1
    imsave(destPath+'/img'+str(cnt)+'.png', i)
