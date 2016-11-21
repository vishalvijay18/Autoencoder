from PIL import Image
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from scipy.misc import toimage
import numpy
import os
import matplotlib.pyplot as plt

path="/home/vishal/Downloads/dataset/"
x_train=numpy.empty([10,19600])
x_test=numpy.empty([10,19600])
i = 0
j = 0
for directory in os.listdir(path):
    for file in os.listdir(path+directory):
        print(path+directory+"/"+file)
        img=Image.open(path+directory+"/"+file)
        featurevector=numpy.array(img).flatten()[:19600] #in my case the images dont have the same dimensions, so [:50] only takes the first 50 values
	if directory == 'train':        
		print str(featurevector.shape)
		x_train[i] = featurevector
		i=i+1
        
	else:
		x_test[j] = featurevector
		j = j+1

print str(x_train.shape)
print "Y list"
print str(x_test.shape)

# this is the size of our encoded representations
encoding_dim = 320  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(19600,))
# add a Dense layer with a L1 activity regularizer
encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
# "encoded" is the encoded representation of the input
encoded = Dense(1280, activation='relu')(input_img)
encoded = Dense(640, activation='relu')(encoded)
encoded = Dense(320, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(640, activation='relu')(encoded)
decoded = Dense(1280, activation='relu')(decoded)
decoded = Dense(19600, activation='sigmoid')(decoded)
#decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)


# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)


# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = (autoencoder.layers[-3])(encoded_input)
decoder_layer = (autoencoder.layers[-2])(decoder_layer)
decoder_layer = (autoencoder.layers[-1])(decoder_layer)
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


from keras.datasets import mnist
import numpy as np
#(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape

autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                nb_epoch=1000,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)

#encoded_imgs[1] = (encoded_imgs[0] * 1.0 + (encoded_imgs[3] * 1.0)) / 2.0

decoded_imgs = decoder.predict(encoded_imgs)



n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(140, 140))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(140, 140))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
