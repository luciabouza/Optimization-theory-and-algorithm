#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def graficar(vector_valsTheta0, vector_valsTheta1, part, theta0, thetaInicial):
    plt.grid(alpha=.4,linestyle='--')
    plt.xlabel('theta[0]')
    plt.ylabel('theta[1]')
    plt.title('Valor de Theta con apha = '+ part)
    plt.plot(vector_valsTheta0, vector_valsTheta1, label='sucesión de puntos')
    plt.plot(theta0[0], theta0[1], marker='o', color="red", label='solución')
    plt.plot(thetaInicial[0], thetaInicial[1], marker='o', color="green", label='punto inicial')
    plt.legend()
    plt.show()


def SGD(A, theta0, thetaInicial, iterations, part):    
    vector_valsTheta0 = list()
    vector_valsTheta1 = list()
    vector_valsTheta0.append(thetaInicial[0])
    vector_valsTheta1.append(thetaInicial[1])
    theta = thetaInicial
        
    for i in range(iterations):

        'determino alpha segun el ejercicio'
        if (part=="0.1"):  alpha = 0.1
        elif (part=="0.01"):  alpha = 0.01 
        elif (part=="0.001"):  alpha = 0.001
        else: alpha = 0.1/(i+1)
        
        'genero vectores y matrices aleatorios w, N, X e y'
        w = np.array([np.random.normal(), np.random.normal()])
        N = np.array([[np.random.normal(), np.random.normal()], [np.random.normal(), np.random.normal()]])        
        X = A + N
        y = np.dot(X, theta0) + w
               
        'hago calculo de desenso'
        direccion = 2*np.dot(np.dot(X.T, X), theta) -2*np.dot(X.T, y)
        theta = theta - alpha * direccion
                
        'guardo valor de theta'
        vector_valsTheta0.append(theta[0])
        vector_valsTheta1.append(theta[1])
   
    return theta, vector_valsTheta0, vector_valsTheta1

####################################
#ejecución e impresión de resultados
####################################

theta0 = np.array([1,1])
A = np.array([[2,1], [1,2]])
Parts = {"0.1", "0.01", "0.001", "decreciente 0.1/(i+1)"} 
thetaInicial = np.array([0, 0])


####### parte C ######

for p in Parts:
    theta, vector_valsTheta0, vector_valsTheta1  = SGD(A, theta0, thetaInicial, 10000, p)
    print ("el resultado con alpha =  ", p, " es: ",theta)
    graficar(vector_valsTheta0, vector_valsTheta1, p, theta0, thetaInicial)


    
    