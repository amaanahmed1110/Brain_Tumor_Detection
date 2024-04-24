import cv2
from keras.models import load_model
from PIL import Image
import numpy as np


model = load_model('BrainTumor10EpochsCategorical.h5')
image = cv2.imread('C:\\Users\\Amaan\\Pycharm-firstprog\\Brain Tumor Project\\pred\\pred6.jpg')

img = Image.fromarray(image)
img = img.resize((64,64))
img = np.array(img)

input_img = np.expand_dims(img, 0)

result = model.predict(input_img)
result_final=np.argmax(result,axis=1)
print(result_final)