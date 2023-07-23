# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 13:04:35 2023

@author: diego
"""
# seleccionar el mejor modelo fully conected usando keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pickle

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Reshape
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




def get_model(lista):
    # de un modelo con la arquitectura especificada
    
    model = Sequential()
    
    # la primera capa solo cambia dimension
    # pasa de n X 28 x 28 (asi viene la imagen) a n * 784
    
    model.add(Reshape((784,), input_shape=(28, 28)))
    
    # primera capa
    model.add(Dense(lista[0], activation = 'relu', input_shape = (784,)))

    for nl in lista[1:]:
        # agreagr las otras capas
        model.add(Dense(nl, activation = 'relu'))

    model.add(Dense(10, activation = 'softmax'))
    
    return model

def todo(lista):
    
    # toma una arquitectura,  crea y entrena el modelo
    # devuelve el modelo y los resutlados de entrenaiento
    
    # crea el modelo
    model = get_model(lista)
    
    # compila
    model.compile(optimizer = 'adam', metrics = 'accuracy',
                  loss = 'sparse_categorical_crossentropy')

    # entrena
    history = model.fit(X_train, y_train, epochs = 70, validation_split = 0.3,
                    callbacks = [EarlyStopping(monitor = 'val_loss', patience = 15)])
    
    # devolver los datos improtantes
    # -1 por que se pasa el ultimo, ( hay uno por cada epoch )
    r1 = history.history['accuracy'][-1]
    r2 = history.history['loss'][-1]
    
    r3 = history.history['val_accuracy'][-1]
    r4 = history.history['val_loss'][-1]
    
    
    # numero de epochs que entreno, recordar el early stopping
    epo  = len(history.history['accuracy'])
    
    return [model, r1, r2, r3, r4, epo]


# todas las opciones que se testearan

opciones = [ [100],
            [200],
            [300],
            [100, 100],
            [300, 100],
            [300, 300],
            [100, 100, 100],
            [300, 300, 250],
            [300, 200, 200, 100],
            [500, 400, 300, 200, 100],
            [500, 400, 300, 300, 300],
            [700, 500, 400, 300, 300],
            [500, 300, 300, 200, 200, 100],
            [500, 400, 400, 300, 300, 200, 100]]


datos = []

# para cada modelo, si es el mejor hasta ahora lo quiero guardar
indx_best = 0  # indice del mejor modelo
best = 0 # val accuracy del mejor modelo

# checar cada de las arquitecturas
for i in range(len(opciones)):
    
    # tomar la arquitectura correspondiente
    
    opcion = opciones[i]
    
    # crear y entrenal modelo
    result = todo(opcion)
    
    # toma el modelo
    temp_model = result[0]
    
    # guardar los datos en formato conceniente
    dicc = {'tipo': str(opcion), 'accuracy': result[1], 
            'loss': result[2], 
            'val_accuracy': result[3],
            'val_loss': result[4],
            'epochs': result[5]}
    
    # agregar este diccionario a una lista
    datos.append(dicc)
    
    # checar si es el mejor hasta ahora, se toma val_accuracy para esto
    # si es el mejor hasta ahora se salva
    if result[3] > best:
        indx_best = i # indice del mejor modelo hasta ahora
        best = result[3] # actualizar
        #temp_model.save('best_keras_dense.h5')
        


# transformar los datos resultantes a un data frame
df = pd.DataFrame(datos)
df = df.set_index('tipo')
# salvar como excel
df.to_csv('Results_keras_dense.csv')

print('La mejor val acc es ' + str(best))
print('Alcanzada con la arquitectura ' + str([784] + opciones[indx_best] + [10]))

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






