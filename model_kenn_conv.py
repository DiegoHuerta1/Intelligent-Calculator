# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 16:39:28 2023

@author: diego
"""

# seleccionar el mejor modelo de conv nn usando keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pickle

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import tensorflow as tf


# usar gpu import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


# cargar y transformar datos

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# no es necesario hacer one.hot encode para y_train
# esto debido a que se usa 'sparse_categorical_crossentropy'
# que espera las etiquetas reales en ese formato


# proponer 4 modelos

# modelo 1
# arquitectura
# 32 filtros 3 x 3
# max Pooling 2 x 2
# Flatten
# Dense 128
# Dense 10

model1 = Sequential()
model1.add(Reshape((28, 28, 1), input_shape=(28, 28)))

model1.add(Conv2D(filters = 32, kernel_size = (3, 3), 
                 input_shape = (28, 28, 1),
                 activation = 'relu'))

model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())

model1.add(Dense(128, activation='relu'))

model1.add(Dense(10, activation='softmax'))

# modelo 2
# arquitectura
# 16 filtros 3 x 3
# 8 filtros 2 x 2
# Flatten
# Dense 128
# Dense 10

model2 = Sequential()
model2.add(Reshape((28, 28, 1), input_shape=(28, 28)))

model2.add(Conv2D(filters = 16, kernel_size = (3, 3), 
                 input_shape = (28, 28, 1),
                 activation = 'relu'))

model2.add(Conv2D(filters = 8, kernel_size = (2, 2), 
                 input_shape = (28, 28, 1),
                 activation = 'relu'))


model2.add(Flatten())

model2.add(Dense(128, activation='relu'))

model2.add(Dense(10, activation='softmax'))

# modelo 3
# arquitectura
# 32 filtros 2 x 2
# 16 filtros 2 x 2
# max Pooling 2 x 2
# Flatten
# Dense 128
# Dense 128
# Dense 10

model3 = Sequential()
model3.add(Reshape((28, 28, 1), input_shape=(28, 28)))

model3.add(Conv2D(filters = 32, kernel_size = (2, 2), 
                 input_shape = (28, 28, 1),
                 activation = 'relu'))

model3.add(Conv2D(filters = 16, kernel_size = (2, 2), 
                 input_shape = (28, 28, 1),
                 activation = 'relu'))

model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(Flatten())

model3.add(Dense(128, activation='relu'))

model3.add(Dense(128, activation='relu'))

model3.add(Dense(10, activation='softmax'))

# modelo 4
# arquitectura
# 32 filtros 3 x 3
# 16 filtros 2 x 2
# max Pooling 2 x 2
# Flatten
# Dense 128
# Dense 10

model4 = Sequential()
model4.add(Reshape((28, 28, 1), input_shape=(28, 28)))

model4.add(Conv2D(filters = 32, kernel_size = (3, 3), 
                 input_shape = (28, 28, 1),
                 activation = 'relu'))

model4.add(Conv2D(filters = 16, kernel_size = (2, 2), 
                 input_shape = (28, 28, 1),
                 activation = 'relu'))

model4.add(MaxPooling2D(pool_size=(2, 2)))

model4.add(Flatten())

model4.add(Dense(128, activation='relu'))

model4.add(Dense(10, activation='softmax'))


def entrenar(model):
    # funcion usada para entrenar modelos y regresar los resultados
    
    # compilar y entrenar
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs = 15, 
                        validation_split = 0.3,
                    callbacks = [EarlyStopping(monitor = 'val_loss', patience = 5)])

    
    # devolver los datos improtantes
    # -1 por que se pasa el ultimo, ( hay uno por cada epoch )
    r1 = history.history['accuracy'][-1]
    r2 = history.history['loss'][-1]
    
    r3 = history.history['val_accuracy'][-1]
    r4 = history.history['val_loss'][-1]
    
    
    # numero de epochs que entreno, recordar el early stopping
    epo  = len(history.history['accuracy'])
    
    return [r1, r2, r3, r4, epo]


# modelos propuestos
modelos = [model1, model2, model3, model4]
datos = []

# para cada modelo, si es el mejor hasta ahora lo quiero guardar
indx_best = 0  # indice del mejor modelo
best = 0 # val accuracy del mejor modelo

for i in range(len(modelos)):
    
    # tomar modelo
    model = modelos[i]
    
    # entrenar modelo
    result = entrenar(model)
    
    # guardar los datos en formato conceniente
    dicc = {'opcion': 'modelo ' + str(i+1), 'accuracy': result[0], 
            'loss': result[1], 
            'val_accuracy': result[2],
            'val_loss': result[3],
            'epochs': result[4]}
    
    # agregar este diccionario a una lista
    datos.append(dicc)
    
    # checar si es el mejor hasta ahora, se toma val_accuracy para esto
    # si es el mejor hasta ahora se salva
    if result[2] > best:
        indx_best = i # indice del mejor modelo hasta ahora
        best = result[2] # actualizar
        #model.save('best_keras_conv.h5')


# transformar los datos resultantes a un data frame
df = pd.DataFrame(datos)
df = df.set_index('opcion')
# salvar como excel
df.to_csv('Results_keras_conv.csv')

print('La mejor val acc es ' + str(best))
print('Alcanzada con la arquitectura del modelo  ' + str(indx_best + 1))


"""
print('Results')
print(history.history['accuracy'][-1])
print(history.history['val_accuracy'][-1])

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'val'])
plt.title('Loss')
plt.show()


plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train', 'val'])
plt.title('Accuracy')
plt.show()
"""






