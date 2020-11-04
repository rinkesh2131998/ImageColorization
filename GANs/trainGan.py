import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Activation, Concatenate, Dropout, Input, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

##############################################################
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True, help='Path to folder with train images')
parser.add_argument('-s', '--split', type=float, default=0.95, help='Train-Test Split(default: 0.95)')
parser.add_argument('-lr', '--learning', type=float, default=0.0002, help='learning rate for Adam(default:0.0002)')
parser.add_argument('-b', '--batchSize', type=int, default=32, help='Batch Size for training(default: 32)')
parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs to train(default: 100)')
parser.add_argument('-bt1', '--beta1', type=float, default=0.5, help='beta1 for Adam(default: 0.5)')
parser.add_argument('-bt2', '--beta2', type=float, default=0.999, help='beta2 for Adam(default: 0.999)')
parser.add_argument('-m', '--model', type=int, default=10, help='Number of epochs after which to save the model(default: 10)')
args = vars(parser.parse_args())

##############################################################
#data path
PATH = args['path']
#train-test split
split = args['split']
#paths to save generator model and output images after some epochs of training
PATH2MODEL = './output/models/'
PATH2IMAGES = './output/images/'
if not os.path.exists(PATH2MODEL):
	os.makedirs(PATH2MODEL)
if not os.path.exists(PATH2IMAGES):
	os.makedirs(PATH2IMAGES)
mSave = args['model'] #save images after evry mSave epochs

##############################################################
#loading images
X = []
for f in os.listdir(PATH):
	x = img_to_array(load_img(PATH+f))
	X.append(x)
X = np.array(X, dtype='float32')
split = int(len(X)*split)
print(split)
##############################################################
#splitting train test and Normalisation
X = 1/255.0 * X
np.random.shuffle(X)
X = 1.0/255.0 * X
Xlab = rgb2lab(X)
np.random.shuffle(Xlab)
XtrainLab = Xlab[:split]
XtestLab = Xlab[split:]
Xtrain = [(XtrainLab[:,:,:,0]/50) - 1, ((XtrainLab[:,:,:,1:]+128.0)/255*2)-1]
Xtrain[0] = Xtrain[0].reshape((-1,256,256,1))
Xtest = [(XtestLab[:,:,:,0]/50) -1, ((XtestLab[:,:,:,1:]+128.0)/255*2)-1]


##############################################################
#defining the model
#define generator
def make_generator(input_shape=(256,256,1)):
  init = tf.keras.initializers.RandomNormal(stddev=0.02)
  f = 64
  def genE(lay_inp, filters_n, bn=True):
    g = Conv2D(filters_n, (4,4), (2,2), padding='same', kernel_initializer=init)(lay_inp)
    if bn:
      g = BatchNormalization(momentum=0.8)(g)
    g = LeakyReLU(0.2)(g)
    return g

  def genD(lay_inp, skip_inp, filters_n, dropout=0.0):
    d = Conv2DTranspose(filters_n, (4,4), (2,2), padding='same', kernel_initializer=init)(lay_inp)
    d = BatchNormalization(momentum=0.8)(d)
    if dropout>0.0:
      d = Dropout(dropout)(d)
    d = Concatenate()([d, skip_inp])
    d = Activation('relu')(d)
    return d

  in_img = Input(shape = input_shape)
  e1 = genE(in_img, f, False)
  e2 = genE(e1, 2*f)
  e3 = genE(e2, 4*f)
  e4 = genE(e3, 8*f)
  e5 = genE(e4, 8*f)
  e6 = genE(e5, 8*f)
  e7 = genE(e6, 8*f)
  m = Conv2D(512,(4,4), strides=2, padding='same', kernel_initializer=init )(e7)
  m = Activation('relu')(m)
  d1 = genD(m, e7, 8*f, 0.5)
  d2 = genD(d1, e6, 8*f, 0.5)
  d3 = genD(d2, e5, 8*f, 0.5)
  d4 = genD(d3, e4, 8*f)
  d5 = genD(d4, e3, 4*f)
  d6 = genD(d5, e2, 8*f)
  d7 = genD(d6, e1, 8*f)
  out_img = Conv2DTranspose(2, (4,4), (2,2), padding='same', kernel_initializer=init)(d7)
  out_img = Activation('tanh')(out_img)
  model = Model(in_img, out_img)
  return model

#define discriminator
def make_discriminator(image_shape=[(256,256,1), (256,256,2)]):
  init = tf.keras.initializers.RandomNormal(stddev=0.02)
  f = 64
  def gen(lay_inp, filters_n, strides=1, bn=True):
    d = Conv2D(filters_n, (4,4), strides = strides, padding='same', kernel_initializer=init)(lay_inp)
    if bn:
      d = BatchNormalization(momentum = 0.8)(d)
    d = LeakyReLU(0.2)(d)
    return d

  img1 = Input(shape = image_shape[0])
  img2 = Input(shape = image_shape[1])

  img_com = Concatenate()([img1, img2])
  d = gen(img_com, f, 2, False)
  d = gen(d, 2*f, 2)
  d = gen(d, 4*f, 2)
  d = gen(d, 8*f, 2)
  d = gen(d, 8*f)
  d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
  outPut = Activation('sigmoid')(d)

  model = Model([img1, img2], outPut)
  opt = Adam(0.0002, 0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
  return model

#making the complete Gan model for training generator
def make_gan(gen, dis, image_shape=(256,256,1)):
  dis.trainable = False
  inp_img = Input(shape = image_shape)
  generatorOutput = gen(inp_img)
  discriminatorOutput = dis([inp_img, generatorOutput])
  model = Model(inp_img, [discriminatorOutput, generatorOutput])
  opt = Adam(0.0002, 0.5)
  model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
  return model


##############################################################
#making the models
generator = make_generator()
discriminator = make_discriminator()
gan = make_gan(generator, discriminator)
print(generator.summary())
print('\n\n\n\n\n')
print(discriminator.summary())
print('\n\n\n\n\n')
print(gan.summary())
print('\n\n\n\n\n')


##############################################################
#training parameters
Lr = args['learning']
Beta1 = args['beta1']
Beta2 = args['beta2']
BATCH_SIZE = args['batchSize']
EPOCHS = args['epochs']
STEPS_PER_EPOCHS = int(len(Xtrain[0])/BATCH_SIZE)
PATCH = (gan.output[0].shape[1])


##############################################################
#utility functions:
def smooth_ones(y):
  return y - 0.3 + (np.random.random(y.shape)*0.5)

def smooth_zeros(y):
  return y  + np.random.random(y.shape) * 0.3

def gen_real_samples(dataset, index, n_samples=BATCH_SIZE, patch_shape=PATCH):
  trainL, trainAB = dataset
  X1, X2 = trainL[index: index+n_samples], trainAB[index: index+n_samples]
  print('Images from: ' + str(index) + ' to: ' + str(index+n_samples), end='\t')
  y = np.ones((n_samples, patch_shape, patch_shape, 1))
  y = smooth_ones(y)
  return [X1, X2], y

def gen_fake_samples(generator, samples, patch_shape=PATCH):
  X = generator.predict(samples)
  y = np.zeros((len(X), patch_shape, patch_shape, 1))
  y = smooth_zeros(y)
  return X, y

def lab_to_img(img):
  canvas = np.zeros((256,256,3))
  for i in range(len(img)):
    for j in range(len(img[i])):
      pix = img[i,j]
      canvas[i,j] = [(pix[0] + 1) * 50,(pix[1] +1) / 2 * 255 - 128,(pix[2] +1) / 2 * 255 - 128]
  canvas = (lab2rgb(canvas)*255.0).astype('uint8')
  return canvas


##############################################################
#define the training
#training funtion
genLoss = []
disLoss1 = []
disLoss2 = []
def train(generator, discriminator, gan, dataset, n_epochs=EPOCHS, n_batch=BATCH_SIZE, n_steps=STEPS_PER_EPOCHS):
  for e in range(1, n_epochs + 1):
    dLoss1 = 0
    dLoss2 = 0
    gLoss = 0
    cnt=0
    for i in range(n_steps):
      [XrealL, XrealAB], yReal = gen_real_samples(dataset,cnt,n_batch)
      XfakeAB, yFake = gen_fake_samples(generator, XrealL)
      dloss1 = discriminator.train_on_batch([XrealL , XrealAB ], yReal )
      dloss2 = discriminator.train_on_batch([XrealL , XfakeAB ], yFake )
      gloss, _, _ = gan.train_on_batch(XrealL, [yReal, XrealAB])
      print( e , '\t',i,'--->', 'disLoss1: ', dloss1, 'disLoss2: ', dloss2,  'genLoss: ', gloss)
      dLoss1+=dloss1
      dLoss2+=dloss2
      gLoss +=gloss
      cnt+=n_batch
      if i!=0 and ( i== int(n_steps/3) or i== int(n_steps/3 * 2) or i==(n_steps-1)):
        n = np.random.randint(0, len(Xtest[0]))
        x = Xtest[0][n]
        y = Xtest[1][n]
        pred = generator.predict(x.reshape((-1, 256, 256, 1)))
        pred = pred.reshape((256,256,2))
        imgPred = lab_to_img(np.dstack((x, pred)))
        imgOrig = lab_to_img(np.dstack((x, y)))
        img = np.hstack((imgOrig, imgPred))
        imsave(PATH2IMAGES+'Iteration_'+str(e)+'_'+str(i)+'.jpg', img)
    if e%mSave==0:
      generator.save(PATH2MODEL + 'gen'+str(e)+'.h5')
    disLoss1.append(dLoss1/n_steps)
    disLoss2.append(dLoss2/n_steps)
    genLoss.append(gLoss/n_steps)


##############################################################
#training the model
train(generator,discriminator, gan, Xtrain)

##############################################################
#plotting the losses
#average losses over the epochs
plt.figure(figsize=(15,10))
plt.plot(disLoss1,label='D1Loss' )
plt.plot(disLoss2,label='D2Loss')
plt.plot(genLoss,label='Genloss' )
plt.title('Average loss over each Epoch')
plt.legend()
plt.show()
