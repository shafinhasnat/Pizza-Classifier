import cv2
import numpy as np
import gzip
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
font=cv2.FONT_HERSHEY_SIMPLEX
n=9950
print('initialize')
pic=6
user1=cv2.imread('test{}.jpg'.format(pic))
user = cv2.resize(user1, (100,100), interpolation = cv2.INTER_AREA)
##cv2.imshow('user',user)
user=user.flatten()
print((user).shape)
loaded_model=joblib.load('SVM_training_model_pizza_notpizza_001.sav')
predict=loaded_model.predict([user])

print(predict)


row,col,chnl=user1.shape

print(abs(col/2))
if predict==[1]:
    color=(0,255,0)
    text='PIZZA'
    place=(int(col/2)-40,35)
    print('Pizza')
else:
    color=(0,0,255)
    text='NO PIZZA'
    place=(int(col/2)-70,35)
    print('Not Pizza')


print('complete')


cv2.rectangle(user1,(0,0),(col,50),color,-1)
cv2.putText(user1,text,place,font,1,(255,255,255),3,cv2.LINE_AA)
cv2.imshow('aqw',user1)
