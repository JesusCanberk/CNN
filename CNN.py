#Importing required libraries
import tensorflow as tf
from tensorflow import keras
from keras import datasets,layers,models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Separating  two sets: train and test sets.
(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()

y_test=y_test.reshape(-1,) #our test matrix becomes nX1 matrix (before it was 1Xn)

#Clarify our classes
img_classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


#With this function we can see our test images
def plot_sample(x,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(x[index])
    plt.xlabel(img_classes[y[index]])


#making normalization
x_test=x_test/255
x_train=x_train/255

#Crating Deep Learning CNN 
deep_learning_model=models.Sequential([
    #It is a first layer Convulation Layer In this part we're exracting the features in images
    layers.Conv2D(filters=32, kernel_size=(3,3),activation="relu",input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),

    #This our second layer Artifical Neural Network Layer. And in this part we will teach our ANN models with features and trainig informations
    layers.Flatten(),#With this code our algortym directly pass the second layer from first layer 
    layers.Dense(64,activation="relu"),
    layers.Dense(10,activation="softmax")
])

deep_learning_model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])



deep_learning_model.fit(x_train,y_train,epochs=5) #It will traing our model with 5 epochs

deep_learning_model.evaluate(x_test,y_test) #It will evaluate our model's performance

y_pred=deep_learning_model.predict(x_test) #Our model makes prediction

y_classes=[np.argmax(element)for element in y_pred] #Extract the class labels from the predictions


print(y_classes[:3])
print(y_test[:3])

plot_sample(x_test,y_test,0)
print(img_classes[y_classes[0]])

plot_sample(x_test,y_test,1)
print(img_classes[y_classes[1]])

plot_sample(x_test,y_test,2)
print(img_classes[y_classes[2]])

plt.show()