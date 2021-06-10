#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time

#####################
#funciones auxiliares
#####################

def graficar(x,y, titulo, xlabel, ylabel):
    plt.grid(alpha=.4,linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    # grafico
    plt.plot(x,y)
    plt.show()
    
def graficarS(x,y, titulo, xlabel, ylabel):
    plt.grid(alpha=.4,linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    # grafico
    plt.scatter(x,y)
    plt.show()
    
    
def eval_f(x):
    return (5*(x[0]**2) + 5*(x[1]**2) + 5*x[0] -3*x[1] -6*x[0]*x[1] + 1.25)

def Proy_Point(x, R):
    return (x/np.linalg.norm(x))*R

def gradF(x):
    return np.array([10*x[0] - 6*x[1] + 5, 10*x[1] - 6*x[0] -3])

def EnConjuntoC(x, R):
    return (x[0]**2 + x[1]**2 <= (R**2))

def EnConjuntoD(x, R):
    return (x[0]**2 + x[1]**2 >= (R**2))
    
def find_s_LineSearch(s, iterations, x, sk, R):
    'seteo mínimos para alpha y el valor mínimo de la función'
    min_Alpha = s
    min_func = eval_f(x)
    'seteo el valor del paso con el que me voy a mover en el segmento [0,s]'
    'para buscar el mejor alpha que minimiza la función'
    step = s/iterations
    for i in range(iterations,0,-1):
        'me muevo, calculo x en ese punto y evalúo la función'
        xTecho = Proy_Point((x - sk * gradF(x)), R)
        direccion = xTecho - x
        x_i = x - (step*i * direccion)
        func = eval_f(x_i)
        'si la función en menor al mínimo hasta el momento,'
        'actualizo min_Alpha y min_func'
        if (func < min_func): 
            min_Alpha = step*i
            min_func = func
    'Devuelvo el mejor alpha'
    return min_Alpha

def gradient_descent(x, iterations, part, epsilon, R, EnConjunto): 
    vector_valorFuncionCosto= list()
    vector_puntosX= list()
    vector_puntosY= list()
    vector_NormaDiferencia = list()
    count =iterations
    sk = 0.001
    
    for i in range(iterations):
      
        'determino sk segun el ejercicio'
        if (part=="Decresciente"):  sk = (1/(i+1))
        else:  sk = find_s_LineSearch(1, 50, x, sk, R)
        
        'hago calculo de desenso'
        valAnterior = x
        valProx = x - sk * gradF(x)
        if (EnConjunto(valProx,R)): #si el punto pertenece al conjunto, no usamos la proyeccion. 
            x = valProx
        else: # si el punto no pertenece al conjunto, debemos proyectar
            x = Proy_Point((valProx), R) #conociendo que alpha=1
        
        'guardo valores a graficar'
        vector_valorFuncionCosto.append(eval_f(x))
        vector_NormaDiferencia.append(np.linalg.norm(x - valAnterior))
        vector_puntosX.append(x[0])
        vector_puntosY.append(x[1])
        
        'si la norma entre el punto anterior y el nuevo es menor a epsilon, finalizo búsqueda'
        if (np.linalg.norm(x - valAnterior) <= epsilon): 
            count = i
            break
   
    return x, vector_valorFuncionCosto, vector_NormaDiferencia, vector_puntosX, vector_puntosY, count
    

#################################################
#ejecución e impresión de resultados ejercicio C
#################################################
#Parts = {"A", "B"} 
Parts = {"Decreciente", "LineSearch"} 
xy_inicial = np.array([2,5]) 

for p in Parts: 
    start_time = time.time()
    x, vector_valorFuncionCosto, vector_NormaDiferencia, vector_puntosX, vector_puntosY, iteraciones  = gradient_descent(xy_inicial, 15, p, 0.0001, 0.25, EnConjuntoC)
    print ("Ejercicio", p , "demoró: ",(time.time() - start_time),"y el resultado fue",x, "en", iteraciones, "iteraciones.")
    graficar(range(len(vector_valorFuncionCosto)),vector_valorFuncionCosto, 'Función de costo vs iteraciones. sk= ' + p, 'iteraciones', 'costo')
    graficar(range(len(vector_NormaDiferencia)),vector_NormaDiferencia, 'Norma de distancia entre puntos de la sucesión vs iteraciones. sk= ' + p, 'iteraciones', 'distancia')
    graficarS(vector_puntosX, vector_puntosY, 'Puntos. sk= ' + p, 'x', 'y')

 
#################################################
#ejecución e impresión de resultados ejercicio D
#################################################

xy_inicial = np.array([0,0]) 
x, vector_valorFuncionCosto, vector_NormaDiferencia, vector_puntosX, vector_puntosY, iteraciones  = gradient_descent(xy_inicial, 4 , 'Decreciente', 0.0001, 1, EnConjuntoD)
print (" Ejericio D1 demoró: ",(time.time() - start_time),"y el resultado fue",x, "en", iteraciones, "iteraciones.")
graficar(range(len(vector_valorFuncionCosto)),vector_valorFuncionCosto, 'Función de costo vs iteraciones D1', 'iteraciones', 'costo')
graficar(vector_puntosX, vector_puntosY, 'Puntos D1', 'x', 'y')
 
xy_inicial = np.array([0.5,0.5]) 
x, vector_valorFuncionCosto, vector_NormaDiferencia, vector_puntosX, vector_puntosY, iteraciones  = gradient_descent(xy_inicial, 4, 'Decreciente', 0.0001, 1, EnConjuntoD)
print (" Ejericio D2 demoró: ",(time.time() - start_time),"y el resultado fue",x, "en", iteraciones, "iteraciones.")
graficar(range(len(vector_valorFuncionCosto)),vector_valorFuncionCosto, 'Función de costo vs iteraciones D2', 'iteraciones', 'costo')
graficarS(vector_puntosX, vector_puntosY, 'Puntos D2', 'x', 'y')
    
   
    
    

