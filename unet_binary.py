from keras.models import *
from keras.layers import Input, Activation, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.layers import Dense, BatchNormalization, ZeroPadding2D, Conv2DTranspose, concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras.backend.tensorflow_backend as K
# from keras import backend as keras
from data_binary import *
from keras.utils import to_categorical


class myUnet(object):

    def __init__(self, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.batch_size = 10
        self.filter_size = 3
        self.channels = 4

    def load_data(self, mask):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data(mask)
        return imgs_train, imgs_mask_train, imgs_test

    def BN_CONV_RELU(self, filters, kernel_size, inputs):
        return Activation(activation='relu')(
            Conv2D(filters, kernel_size, strides=(1, 1), padding='valid',
                   kernel_initializer='glorot_uniform', bias_initializer='zeros')(
                ZeroPadding2D(padding=(1, 1))(  # P=(F-1)/2=(3-1)/2=1
                    BatchNormalization()(inputs))))

    def BN_UPCONV_RELU(self, filters, kernel_size, inputs):
        return Activation(activation='relu')(
            Cropping2D(cropping=((1, 0), (0, 1)))(  # crop top row and right column
                Conv2DTranspose(filters, kernel_size, strides=(2, 2), padding='valid',
                                kernel_initializer='glorot_uniform', bias_initializer='zeros')(
                    BatchNormalization()(inputs))))

    def DOWN(self, channels, filter_size, inputs):
        return self.BN_CONV_RELU(channels, filter_size,
                                 self.BN_CONV_RELU(channels, filter_size, inputs))

    def UP(self, channels, filter_size, inputs, downinput):
        return self.BN_UPCONV_RELU(channels, filter_size,
                                   self.BN_CONV_RELU(channels, filter_size,
                                                     concatenate([inputs, downinput], axis=3)))

    def get_unet(self):
        inputs = Input(shape=(self.img_rows, self.img_cols, 2))  # data_format=(batch, height, width, channels)

        down1 = self.DOWN(self.channels, self.filter_size, inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(down1)
        print("down1 shape:", pool1.shape)
        down2 = self.DOWN(self.channels, self.filter_size, pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(down2)
        print("down2 shape:", pool2.shape)
        down3 = self.DOWN(self.channels, self.filter_size, pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(down3)
        print("down3 shape:", pool3.shape)
        #down4 = self.DOWN(self.channels, self.filter_size, pool3)
        #pool4 = MaxPooling2D(pool_size=(2, 2))(down4)
        #print("down4 shape:", pool4.shape)
        #down5 = self.DOWN(self.channels, self.filter_size, pool4)
        #pool5 = MaxPooling2D(pool_size=(2, 2))(down5)
        #print("down5 shape:", pool5.shape)

        #
        bcr6 = self.BN_CONV_RELU(self.channels, self.filter_size, pool3)
        bcr6 = self.BN_CONV_RELU(self.channels, self.filter_size, bcr6)
        bur6 = self.BN_UPCONV_RELU(self.channels, self.filter_size, bcr6)
        print("bur6 shape:", bur6._keras_shape)
        #

        #up5 = self.UP(self.channels, self.filter_size, bur6, down5)
        #print("up5 shape:", up5._keras_shape)
        #up4 = self.UP(self.channels, self.filter_size, up5, down4)
        #print("up4 shape:", up4._keras_shape)
        up3 = self.UP(self.channels, self.filter_size, bur6, down3)
        print("up3 shape:", up3._keras_shape)
        up2 = self.UP(self.channels, self.filter_size, up3, down2)
        print("up2 shape:", up2._keras_shape)

        concat11 = concatenate([up2, down1], axis=3)
        bcr11 = self.BN_CONV_RELU(self.channels, self.filter_size, concat11)
        bcr11 = self.BN_CONV_RELU(self.channels, self.filter_size, bcr11)
        print("up1 shape:", bcr11.shape)

        convout11 = Conv2D(1, 1, strides=(1, 1), padding='valid', activation='sigmoid')(bcr11)
        print("convout shape:", convout11.shape)
        #sigmoid = Activation(activation='sigmoid')(convout11)
        #print("sigmoid shape:", sigmoid.shape)
        print()

        model = Model(inputs=inputs, outputs=convout11)
        model.summary()

        model.compile(optimizer=Adam(lr=1e-2), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")

        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
        print('Fitting model...')

        model.fit(imgs_train, imgs_mask_train, batch_size=self.batch_size, epochs=10, verbose=1, validation_split=0.2,
                  shuffle=True, callbacks=[model_checkpoint, tensorboard])

        #print('predict test data')
        #imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=2)
        #np.save('results/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("array to image")
        imgs = np.load('results/imgs_mask_test.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save("results/{}.jpg".format(i))


if __name__ == '__main__':
    myunet = myUnet(128,128)
    #myunet.get_unet()
    myunet.train()
    #myunet.save_img()
