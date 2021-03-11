import tensorflow as tf      #importa la libreria de tensorflow para realizar el modelo 
from tensorflow import keras #importa la libreria de keras para realizar el modelo 
import pandas as pd          #importa la libreia pandas necesaria para abrir los datos
import numpy as np           #importa la libreia pandas necesaria para abrir los datos


#========================== datos del modelo ================================== 

datos_train=pd.read_csv('train.csv',header=0)   #carga los datos de entrenamiento
datos_test=pd.read_csv('test.csv',header=0)     #carga los datos de validacion


train_labels=datos_train.pop('Class')      #separa las etiquetas de los datos de entrenamiento 
train_data=datos_train

test_labels=datos_test.pop('Class')        #separa las etiquetas de los datos de validacion 
test_data=datos_test





train_data=train_data/5 #se ajustan los datos para que esten entre 0 y 1 
test_data=test_data/5





#=========================== clasificaciones  =================================
class_names=['Move-Forward','Slight-Right-Turn',
             'Sharp-Right-Turn','Slight-Left-Turn']


#============================ Se crea el modelo ===============================


model=keras.Sequential([
    keras.layers.Flatten(input_shape=(1,4)),  #input layer
    keras.layers.Dense(26,activation= 'sigmoid'), #hidden layer
    keras.layers.Dense(4,activation='softmax')   #output layer
    ])





#============================ Se compila el modelo ============================



model.compile(optimizer='adam',                           #se seleciona el optimizador del modelo
              loss='sparse_categorical_crossentropy',     #se seleciona la funcion de perdida
              metrics=['accuracy'])                       #se pide el dato de precision 





#========================== Se entrena el modelo con los datos ================


model.fit(train_data,train_labels,epochs=100)   #se entrena el modelo con los datos
                                                #se seleciona la cantidad de iteraciones





#=========================== Se prueba el modelo con los datos ================


test_loss,test_acc= model.evaluate(test_data, test_labels, verbose=1)   #se prueba el modelo y se imprime la precision  
print('Test accuracy:', test_acc)










