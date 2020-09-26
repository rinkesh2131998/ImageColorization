import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
from skimage.transform import resize
import numpy as np
import os
import tensorflow as tf
import argparse

#CLI ARGS
parser = argparse.ArgumentParser(prog="GrayScale2RGB", description='Output Script')
parser.add_argument('--path', type=str, default='./dataset/test/', help='testing images directory(default: ./dataset/test)')
parser.add_argument('--premodel', type=str, default='./models/model.h5', help ='Saved model(default directory: ./models/model.h5)')
parser.add_argument('--output',type=str, default='./OutputImages/',help='Output directory to saved images(default: ./OutputImages/)')
args = parser.parse_args()

path = args.path
premodel = args.premodel
output_path = args.output
#print(premodel)
#importing dataset
def img_import(path):
    X = []
    for file in os.listdir(path):
        x = img_to_array(load_img(path + file))
        x = resize(x, (240,288), mode='symmetric')
        X.append(x)
    X = np.array(X,dtype=float)
    if X.shape[3]==1:
        X = 1.0/255*X[:,:,:,0]
    else:
        X = rgb2lab(1.0/255*X)[:,:,:,0]
    X = X.reshape(X.shape+(1,))
    return X

def out_img(output, X, output_path):
    # Output colorizations
    for i in range(len(output)):
      cur = np.zeros((240,288, 3))
      cur[:,:,0] = X[i][:,:,0]
      cur[:,:,1:] = output[i]
      file_name= output_path+ "{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
      imsave(file_name, lab2rgb(cur))


if __name__ == '__main__':
    X = img_import(path)
    #model = kerasModel()
    #model = load_model(premodel)
    output = model.predict(X)
    output = output*128
    out_img(output, X, output_path)
