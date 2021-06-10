# -*- coding: utf-8 -*-

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

#################################################
#####SOLUCION CERRADA ECUACIONES NORMALES #######
##### (Xt.X)-1.Xt.y #############################
#################################################
SolucionCerrada = np.dot( np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), B)
print ("Resultado ecuaciones normales: ")
print (SolucionCerrada)

######################################################
#####DESCENSO POR GRADIENTE CON FUNCION LINEAL #######
######################################################

#####################
#funciones auxiliares
#####################

def graficar(x,y, Parte):
    plt.grid(alpha=.4,linestyle='--')
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo relativo')
    plt.title('Costo Relativo en función iteraciones Parte ' + Parte)
    # grafico
    plt.plot(x,y)
    plt.show()
    
def eval_f(A,b, x):
    return np.linalg.norm(np.dot(A, x) -b)**2
    
def find_Alpha_LineSearch(s, iterations, A, b, x, error, m):
    'seteo mínimos para alpha y el valor mínimo de la función'
    min_Alpha = s
    min_func = eval_f(A,b, x)
    'seteo el valor del paso con el que me voy a mover en el segmento [0,s]'
    'para buscar el mejor alpha que minimiza la función'
    step = s/iterations
    for i in range(iterations,0,-1):
        'me muevo, calculo x en ese punto y evalúo la función'
        x_i = x - (step*i * (1/m) * np.dot(A.T, error))
        func = eval_f(A,b, x_i)
        'si la función en menor al mínimo hasta el momento,'
        'actualizo min_Alpha y min_func'
        if (func < min_func): 
            min_Alpha = step*i
            min_func = func
    'Devuelvo el mejor alpha'
    return min_Alpha


def find_Alpha_Armijo(betha, A, b, x, error, m):
    'inicializo variables de rango de busqueda, factor y sigma'
    s, p = 1, 0
    sigma = 0.1
    'inicializo variables necesarias para validar la desigualdad'
    grad = (1/m) * np.dot(A.T, error)  
    func = eval_f(A,b, x)
    
    while (True):
        'me muevo, calculo x en ese punto y evalúo la función'
        p = p +1
        x_i = x - ((betha**p)* s * grad)
        func_betha = eval_f(A,b, x_i)
        condition = sigma * (betha**p)* s *  np.linalg.norm(np.dot(grad.T,grad))**2
        'si se da la siguiente desigualdad, encontré un punto donde la función disminuye'
        'entonces finalizo búsqueda'
        if ( func_betha - func <= condition): break
    'cuando salimos del while, devuelvo el paso encontrado'
    return (betha**p)*s

def gradient_descent(A, b, x, iterations, solution, part, epsilon):
    m = A.size
    vector_errorRelativo = list()
    count =1000
    
    for i in range(iterations):
        'calculos previos'
        error = np.dot(A, x) - b
      
        'determino alpha segun el ejercicio'
        if (part=="A"):  alpha = 1/(2*(np.linalg.norm(A))**2)
        elif (part=="B"):  alpha = 0.001 * (1/(i+1))
        elif (part=="C"):  alpha = find_Alpha_LineSearch(0.01, 50, A, b, x, error, m)
        else: alpha = find_Alpha_Armijo(0.001, A, b, x, error, m)
        
        'hago calculo de desenso'
        direccion = (1/m) * np.dot(A.T, error)
        x = x - (alpha * direccion) 
        
        'guardo error relativo'
        errorRelativo = np.linalg.norm(solution - x)/np.linalg.norm(solution)
        vector_errorRelativo.append(errorRelativo)
        
        'si la dirección es menor a epsilon, finalizo búsqueda'
        if (np.linalg.norm(direccion) <= epsilon): 
            count = i
            break
   
    return x, vector_errorRelativo, count
    

####################################
#ejecución e impresión de resultados
####################################
Parts = {"A", "B", "C", "D"} 
x_inicial = np.array([1,1]) 

for p in Parts: 
    start_time = time.time()
    x, vector_errorRelativo, iteraciones  = gradient_descent(A, B, x_inicial, 1000, SolucionCerrada, p, 0.0001)
    print ("Parte", p , "demoró: ",(time.time() - start_time),"y el resultado fue",x, "en", iteraciones, "iteraciones.")
    graficar(range(len(vector_errorRelativo)),vector_errorRelativo, p)

