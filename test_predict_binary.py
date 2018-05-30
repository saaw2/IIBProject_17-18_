from unet_binary import *
from data_binary import *
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import scipy.misc

mydata = dataProcess(128,128)

print("loading data")
imgs_test, labels_test = mydata.load_test_data(2)
print("loading data done")

myunet = myUnet(128,128)
model = myunet.get_unet()
print("got unet")

model.load_weights('unet_best.hdf5')
print("loaded model weights")

imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
loss, acc = model.evaluate(imgs_test, labels_test, batch_size=1, verbose=1)
print('loss on test data is ', loss)
print('acc on test data is ', acc)
print("saving test masks")
np.save('npydata/imgs_mask_test.npy', imgs_mask_test)

print("array to image")
imgs_true = labels_test
print('loading done, now saving')
for i in range(imgs_true.shape[0]):
    img = imgs_test[i,:,:,0]
    img_pred = imgs_mask_test[i,:,:,0]
    img_true = imgs_true[i,:,:,0]
    scipy.misc.imsave("results_other/{}_img.jpg".format(i), img)
    scipy.misc.imsave("results_other/{}_pred.jpg".format(i), img_pred)
    scipy.misc.imsave("results_other/{}_true.jpg".format(i), img_true)

"""    
imgs = np.load('imgs_mask_test.npy')
for i in range(imgs.shape[0]):
    img = imgs[i]
    np.save("test_results/{}.npy".format(i), imgs_mask_test)
    #img = array_to_img(img)
    #img.save("test_results/{}.jpg".format(i))

"""