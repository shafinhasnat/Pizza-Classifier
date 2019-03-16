import cv2
import glob
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib
pizza=[]
notpizza=[]


print('loading dataset...')


notpizzapath= "D:\PizzaClassifier\h\*.*"
pizzapath = "D:\PizzaClassifier\pizza\*.*"
for file in glob.glob(pizzapath):
    imp=cv2.imread(file)
    imp = cv2.resize(imp, (100,100), interpolation = cv2.INTER_AREA)
##    cv2.imshow('imp',imp)
    imp=np.array((imp))
    imp=imp.flatten()
    pizza.append(imp)

pizza=np.array(pizza)
print('shape of individual pizza',pizza[0].shape)
print('length of pizza',len(pizza))
pizzalabel=np.ones(len(pizza),dtype=np.uint8)

for f in glob.glob(notpizzapath):
    imn=cv2.imread(f)
    imn = cv2.resize(imn, (100,100), interpolation = cv2.INTER_AREA)
##    cv2.imshow('imn',imn)
    imn=np.array((imn))
    imn=imn.flatten()
    notpizza.append(imn)

    
##cv2.imshow('asd',pizza[0])
notpizza=np.array(notpizza)
print('shape of individual not pizza',notpizza[0].shape)
print('length of not pizza',len(notpizza))
notpizzalabel=np.zeros(len(notpizza),dtype=np.uint8)
##print((pizzalabel).dtype)

trainingImages=np.concatenate((pizza,notpizza))
print('length of training image',len(trainingImages),trainingImages.dtype,trainingImages.ndim)

traininglabel=np.append(pizzalabel,notpizzalabel)
print('length of training label',len(traininglabel),traininglabel.dtype,traininglabel.ndim)

##print(traininglabel[397:])
print('loading dataset successful!!!')

print('training initialize...')
x_train,x_test,y_train,y_test=train_test_split(trainingImages,traininglabel,test_size=0,random_state=42)
##
classifier=svm.SVC(kernel='linear')
classifier.fit(trainingImages,traininglabel)
print('training successful!!!')
##
filename='SVM_training_model_pizza_notpizza_001.sav'
joblib.dump(classifier,filename)
print('successfully created training file!!!')
##cv2.imshow('asdasdasd',trainingImages[1232])
