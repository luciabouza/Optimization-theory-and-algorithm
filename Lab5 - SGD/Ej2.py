#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def graficar(valoresCorrectosTest, valoresPredecidosTest, valoresCorrectosTrain, valoresPredecidosTrain):
    
    plt.grid(alpha=.4,linestyle='--')
    plt.xlabel('Muestra')
    plt.ylabel('valores Predecidos')
    plt.title('Valores predecidos para las muestras de Test (o gatos, x conejos)')
       

    for i in range(len(valoresCorrectosTest)):
        if valoresCorrectosTest[i] == 0: # es gato en realidad
            plt.plot(i, valoresPredecidosTest[i], marker='o', color="red")
        else: #Es conejo en realidad
            plt.plot(i, valoresPredecidosTest[i], marker='x', color="red")
  
    plt.show()
    
    plt.grid(alpha=.4,linestyle='--')
    plt.xlabel('Muestra')
    plt.ylabel('valores Predecidos')
    plt.title('Valores predecidos para las muestras de Train (o gatos, x conejos)')
    
    for i in range(len(valoresCorrectosTrain)):
        if valoresCorrectosTrain[i] == 0: # es gato en realidad
            plt.plot(i, valoresPredecidosTrain[i], marker='o', color="green")
        else: #Es conejo en realidad
            plt.plot(i, valoresPredecidosTrain[i], marker='x', color="green")
    
    plt.show()


def SGD(X_train, y_train, theta0, iterations, epsilon):    
    theta = theta0
    alpha = pow(10, -9)
    direccion = np.array(theta.shape) 
    
    for i in range(iterations):
        for j in range(X_train.shape[1]):
            image = X_train[:,j]
            
            'hago calculo de desenso'
            ax = np.dot(theta.T,image)   
            aTxxT = np.dot(ax, image.T)
            yXt = y_train[j]* image.T           

            
            if(ax < 0):
                direccion = 2*(epsilon**2)*aTxxT -2*epsilon*yXt
            if(ax >= 0): 
                direccion = 2*aTxxT -2*yXt
            if (ax == 0 ):
                direccion = -(epsilon + 1)*yXt

            theta = theta - alpha * direccion
            
    return theta


####################################################################
####### cargo datos y creo conjuntos de entrenamiento y testeo #####
####################################################################

conejos = np.loadtxt(fname = './Conejos.asc')
gatos = np.loadtxt(fname = './Gatos.asc')

conejos_train = conejos[0:conejos.shape[0],0:20]
gatos_train = gatos[0:gatos.shape[0],0:20]

conejos_test = conejos[0:conejos.shape[0],20:30]
gatos_test = gatos[0:gatos.shape[0],20:30]

X_train = np.concatenate((conejos_train, gatos_train), axis=1)
X_test = np.concatenate((conejos_test, gatos_test), axis=1)

y_train = np.concatenate((np.ones(20), np.zeros(20)))
y_test = np.concatenate((np.ones(10), np.zeros(10)))


####################################################################
############## entrenamiento #######################################
####################################################################

'inicializo vector inicial de pesos'
tamaño_imagen = X_train.shape[0]
a0 = np.zeros(tamaño_imagen)
a0[-1] = 1

a = SGD(X_train, y_train, a0, 69, epsilon = 0.1)
print ("el resultado de los pesos es: ",a)


####################################################################
############## evaluación ##########################################
####################################################################


y_predictTest = np.dot(a.T, X_test)
y_predictTrain = np.dot(a.T, X_train)

graficar(y_test, y_predictTest, y_train, y_predictTrain)

valspredictTest = np.zeros(20)
valspredictTrain = np.zeros(40)
valspredictTest[y_predictTest>0] = 1
valspredictTrain[y_predictTrain>0] = 1

print('###### metricas para testing ########')
print(classification_report(y_test, valspredictTest))

print('####### metricas para train ########')
print(classification_report(y_train, valspredictTrain))

print('####### metricas para todo el conjunto ########')
print(classification_report(np.concatenate((y_train, y_test)), np.concatenate((valspredictTrain, valspredictTest))))
 
