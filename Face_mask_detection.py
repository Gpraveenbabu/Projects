!pip install kaggle
!mkdir -p~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d omkargurav/face-mask-dataset
from zipfile import ZipFile
data='/content/face-mask-dataset.zip'
with ZipFile(data,'r') as zip:
  zip.extractall()
  print('done')
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.image as mpimg
from google.colab.patches import cv2_imshow
from PIL import Image
from sklearn.model_selection import train_test_split
with_mask=os.listdir('/content/data/with_mask')
without_mask=os.listdir('/content/data/without_mask')
print(with_mask[:5])
print(without_mask[:5])
print('Number of Images with mask',len(with_mask))
print('Number of Images without mask',len(without_mask))
with_mask_labels=[1]*3725
without_mask_labels=[0]*3828
print(with_mask_labels[0:5])
print(without_mask_labels[0:5])
print(len(with_mask_labels))
print(len(without_mask_labels))
labels=with_mask_labels+without_mask_labels
print(len(labels))
#Having Mask
img=mpimg.imread('/content/data/with_mask/with_mask_3478.jpg')
imgplot=plt.imshow(img)
plt.show()
#Not Having Mask
img=mpimg.imread('/content/data/without_mask/without_mask_789.jpg')
imgplot=plt.imshow(img)
plt.show()
#converting images to numpy array
with_mask_path='/content/data/with_mask'
data=[]
for img_file in with_mask:
  image=Image.open(with_mask_path+'/'+img_file)
  image=image.resize((128,128))
  image=image.convert('RGB')
  image=np.array(image)
  data.append(image)

without_mask_path='/content/data/without_mask'

for img_file in without_mask:
  image=Image.open(without_mask_path+'/'+img_file)
  image=image.resize((128,128))
  image=image.convert('RGB')
  image=np.array(image)
  data.append(image)
type(data)
len(data)
type(data[0])
data[0].shape
#converting image list and label list to numpy array
x=np.array(data)
y=np.array(labels)
type(x)
type(y)
print(x.shape)
print(y.shape)
print(y)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x.shape)
print(X_train.shape)
print(X_test.shape)
X_train_scaled=X_train/255
X_test_scaled=X_test/255
X_train_scaled[0]
import tensorflow as tf
from tensorflow import keras
num_of_classes=2
model=keras.Sequential()
model.add(keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_of_classes,activation='sigmoid'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
#now running the neural network
history=model.fit(X_train_scaled,Y_train,validation_split=0.1,epochs=5)
loss,accuracy=model.evaluate(X_test_scaled,Y_test)
print('Test Accuracy',accuracy)
h=history

#plot of loss value
plt.plot(h.history['loss'],label='train loss')
plt.plot(h.history['val_loss'],label='validation loss')
plt.legend()
plt.show()


#plot of accuracy value
plt.plot(h.history['acc'],label='train accuracy')
plt.plot(h.history['val_acc'],label='validation accuracy')
plt.legend()
plt.show()
#predictive system
input_image_path=input('Path of the image to be predicted')
input_image=cv2.imread(input_image_path)
cv2_imshow(input_image)
input_image_resized=cv2.resize(input_image,(128,128))
input_image_scaled=input_image_resized/255
image_reshaped=np.reshape(input_image_scaled,[1,128,128,3])
input_prediction=model.predict(image_reshaped)
print(input_prediction)
input_pred_label=[np.argmax(input_prediction)]
print(input_pred_label)
if input_pred_label==1:
  print('Person in the image was not wearing the mask')
else:
  print('Person in the image was wearing the mask')

