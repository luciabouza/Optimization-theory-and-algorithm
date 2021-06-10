#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:08:06 2020

@author: luciabouza
"""
import numpy as np
import matplotlib.pyplot as plt
import time

#####################################
###### VECTORES A Y B ###############
#####################################
A =  np.array ([[-4.100000000000000000e+01, 2.000000000000000000e+01],
[-4.600000000000000000e+01, -8.000000000000000000e+00],
[-5.000000000000000000e+00, -3.300000000000000000e+01],
[-5.500000000000000000e+01, 1.000000000000000000e+00],
[-5.500000000000000000e+01, -6.000000000000000000e+00]])

B =  np.array([ 8.000000000000000000e+00, 5.000000000000000000e+00, -3.000000000000000000e+00, 1.000000000000000000e+01, 4.000000000000000000e+00])


def graficar(x,y):
    plt.grid(alpha=.4,linestyle='--')
    plt.xlabel('Iteraciones')
    plt.ylabel('f')
    plt.title('Valor de la función, en función iteraciones')
    plt.plot(x,y)
    plt.show()
    
def eval_func(A,b, x, lambdaa):  
    return 0.5*np.linalg.norm(np.dot(A, x) -b)**2 + lambdaa*np.linalg.norm(x, ord=1)


def Soft_Thresholding(x, lambdaa):
    xAux = x
    xAux[abs(xAux)<=lambdaa] = 0
    xAux[xAux> lambdaa] -= lambdaa
    xAux[xAux< -lambdaa] += lambdaa
    return xAux

def proximal_f(A, b, zu, rho):
    term1 = np.linalg.inv(np.dot(A.T, A) + rho*np.identity(A.T.shape[0]))
    term2 = np.dot(A.T,b) + zu
    return np.dot(term1,term2)

def ADMM(A, b, x, z, u, iterations, epsilon, lambdaa, rho):
    vector_valsFuncion = list()
    vector_valsFuncion.append(eval_func(A,b, x, lambdaa))
    
    for i in range(iterations):
        x = proximal_f(A, b, (z - u), rho)
        z = Soft_Thresholding((x + u), lambdaa)
        u = u + x -z
        
        'guardo valor de la funcion'
        vector_valsFuncion.append(eval_func(A,b, x, lambdaa))
    
        'si la diferencia de las funciones en los puntos es menor a epsilon, finalizo búsqueda'
        if (abs(vector_valsFuncion[-2] - vector_valsFuncion[-1]) <= epsilon): 
            break
        
    return x, vector_valsFuncion
      

def Proximal_gradient_descent(A, b, x, iterations, epsilon, lambdaa):    
    m = A.size
    vector_valsFuncion = list()
    vector_valsFuncion.append(eval_func(A,b, x, lambdaa))
    
    for i in range(iterations):
        'calculos previos'
        error = np.dot(A, x) - b 

        alpha = 1/(np.linalg.norm(np.dot(A.T,A)))
        
        'hago calculo de desenso'
        direccion = (0.5/m) * np.dot(A.T, error)
        xDescenso = x - (lambdaa * alpha * direccion) 
        x = Soft_Thresholding(xDescenso, lambdaa)
                
        'guardo valor de la funcion'
        vector_valsFuncion.append(eval_func(A,b, x, lambdaa))
              
        'si la diferencia de las funciones en los puntos es menor a epsilon, finalizo búsqueda'
        if (abs(vector_valsFuncion[-2] - vector_valsFuncion[-1]) <= epsilon): 
            break
   
    return x, vector_valsFuncion

####################################
#ejecución e impresión de resultados
####################################
x_inicial = np.array([1,1]) 
z_inicial = np.array([1,1]) 
u_inicial = np.array([1,1]) 
lambdaa = 0.15
rho = 0.01

####### parte A ######
start_time = time.time()
x, vector_valsFuncion  = Proximal_gradient_descent(A, B, x_inicial, 1000, 0.0001, lambdaa)
print ("el resultado con Proximal gradient descent es: ",x, " demoró: ",(time.time() - start_time))
graficar(range(len(vector_valsFuncion)),vector_valsFuncion)

####### parte B ######
start_time = time.time()
x, vector_valsFuncion  = ADMM(A, B, x_inicial, z_inicial, u_inicial, 1000, 0.0001, lambdaa, rho)
print ("el resultado con ADMM es: ",x, " demoró: ",(time.time() - start_time))
graficar(range(len(vector_valsFuncion)),vector_valsFuncion)
    
    