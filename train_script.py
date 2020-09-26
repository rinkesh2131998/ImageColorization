import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
from skimage.transform import resize
import numpy as np
import os
import tensorflow as tf
import argparse

#CLI arguments
parser = argparse.ArgumentParser(prog="GrayScale2RGB", description='Train model on Custom dataset')
parser.add_argument('--path', type=str, default='./dataset/train/', help='dataset directory(default: ./dataset/train)')
parser.add_argument('--model', type=bool, default=False ,help='Pretrained model weights directory(default: False, If true then model in ./models is used)')
parser.add_argument('--premodel', type=str, default='./models/model.h5', help ='The saved weights to train(default directory: ./models/model.h5)')
parser.add_argument('--split', type=float, default=0.8, help='Split ratio for training and testing(default: 0.8)')
parser.add_argument('--batch_size', type=int, default=15, help='Batch Size for training(default: 15)')
parser.add_argument('--epochs', type=int, default=30, help='no. of epochs for training(default: 30)')
parser.add_argument('--checkpoint', type=str, default='./models/', help='location where the model is saved(default: ./models)')
parser.add_argument('--epochSteps', type=int, default=30, help='step per epochs for training(default: 50)')

args = parser.parse_args()

path = args.path
pre_chk = args.model
pretrained_model= args.premodel
checkpoint = args.checkpoint
split = args.split
batch_size = args.batch_size
epochs = args.epochs

#print(path, pretrained_model, checkpoint,split,batch_size,epochs)


#importing dataset
def img_import(path):
    X = []
    for file in os.listdir(path):
        x = img_to_array(load_img(path + file))
        x = resize(x, (240,288), mode='symmetric')
        X.append(x)
    X = np.array(X,dtype=float)
    return X

#train-test split
def train_test_split(X,split=0.9):
    split = int(split*(len(X)))
    #print(split)
    Xtrain = X[:split]
    Xtrain = 1.0/255*Xtrain
    Xtest = X[split:]
    Xtest = 1.0/255*Xtest
    #preparing testing images
    Xtest = rgb2lab(Xtest)
    Xtest = Xtest[:,:,:,0]
    Xtest = Xtest.reshape(Xtest.shape+(1,))
    ytest = rgb2lab(1.0/255*X[split:])
    ytest = ytest[:,:,:,1:]/128
    #get size of input image
    #img_rows = X.shape[1]
    #img_cols = X.shape[2]
    return Xtrain, Xtest, ytest

def image_a_b_gen(Xtrain,batch_size=15):
    # Changing training image orientation,size etc
    imgdata = ImageDataGenerator(
            shear_range=0.3,
            zoom_range=0.3,
            rotation_range=30,
            horizontal_flip=True)
    for batch in imgdata.flow(Xtrain, batch_size=batch_size):
        lab = rgb2lab(batch)
        X_batch = lab[:,:,:,0]
        Y_batch = lab[:,:,:,1:] / 128 #Lab images have intesity -128 to 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

def kerasModel(img_rows=240, img_cols=288):
    enc_input = Input(shape=(img_rows,img_cols ,1))
    model_enc = Conv2D(filters=64, kernel_size=3,  padding='same', activation='relu',strides = 2)(enc_input)
    model_enc = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(model_enc)
    model_enc = BatchNormalization()(model_enc)
    model_enc = Conv2D(128, (3, 3), activation='relu', padding='same')(model_enc)
    model_enc = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(model_enc)
    model_enc = BatchNormalization()(model_enc)
    model_enc = Conv2D(256, (3, 3), activation='relu', padding='same')(model_enc)
    model_enc = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(model_enc)
    model_enc = BatchNormalization()(model_enc)
    model_enc = UpSampling2D(size=(2,2))(model_enc)
    model_enc = Conv2D(512, (3, 3), activation='relu', padding='same')(model_enc)
    model_enc = Conv2D(512, (3, 3), activation='relu', padding='same')(model_enc)
    model_enc = BatchNormalization()(model_enc)
    model_enc = UpSampling2D(size=(4,4))(model_enc)
    model_enc = Conv2D(256, (3, 3), activation='relu', padding='same')(model_enc)
    model_enc = Conv2D(256, (3, 3), activation='relu', padding='same')(model_enc)
    model_enc = Conv2D(256, (3, 3), activation='relu', padding='same')(model_enc)
    model_enc = BatchNormalization()(model_enc)
    model_enc = UpSampling2D(size=(2,2))(model_enc)
    outputs = Conv2D(2, (1, 1), activation='tanh', padding='same')(model_enc)
    model = Model(inputs=enc_input, outputs=outputs)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    print(model.summary())
    return model

if __name__ == '__main__':
    Xtrain, Xtest, ytest = train_test_split(img_import(path), split)
    model = kerasModel()
    tf_path = checkpoint+'tensorboard/'
    t_board = TensorBoard(log_dir=tf_path)
    m_path = checkpoint + 'modelRgb.h5'
    m_chk_point = ModelCheckpoint(m_path, monitor ='loss', verbose=1, save_best_only=True, mode='min')
    if pre_chk:
        model.load_weights(pretrained_model)
    model.fit(image_a_b_gen(Xtrain, batch_size), callbacks=[m_chk_point, t_board], epochs = epochs, steps_per_epoch = 50)
    print(model.evaluate(Xtest, ytest,batch_size=batch_size))
