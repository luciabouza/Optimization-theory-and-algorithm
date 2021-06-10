#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#####################################
###### matrices A, B y C. ###############
###### vectores alpha, betha y gamma. ###############
#####################################

A = np.array ([[2.000000000000000000e+00, 2.000000000000000000e+00, 2.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 2.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 0.000000000000000000e+00],
[0.000000000000000000e+00, 2.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 2.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 2.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00],
[0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00, 3.000000000000000000e+00],
[1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 2.000000000000000000e+00, 1.000000000000000000e+00, 0.000000000000000000e+00, 3.000000000000000000e+00, 0.000000000000000000e+00],
[1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 4.000000000000000000e+00, 1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00],
[0.000000000000000000e+00, 2.000000000000000000e+00, 1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00, 2.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 2.000000000000000000e+00],
[2.000000000000000000e+00, 1.000000000000000000e+00, 2.000000000000000000e+00, 2.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 2.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00],
[1.000000000000000000e+00, 2.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00, 0.000000000000000000e+00, 3.000000000000000000e+00],
[3.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 2.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00, 2.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00],
[1.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00, 0.000000000000000000e+00, 3.000000000000000000e+00, 2.000000000000000000e+00]])
 
alpha = np.array ([-2.000000000000000000e+00,
-4.000000000000000000e+00,
-5.000000000000000000e+00,
-4.000000000000000000e+00,
7.000000000000000000e+00,
-9.000000000000000000e+00,
-6.000000000000000000e+00,
-9.000000000000000000e+00,
-7.000000000000000000e+00,
-3.000000000000000000e+00])

B = np.array([[-3.000000000000000000e+00, -1.000000000000000000e+00, 5.000000000000000000e+00, -5.000000000000000000e+00, -9.000000000000000000e+00, 3.000000000000000000e+00, -2.000000000000000000e+00, 1.000000000000000000e+00, -9.000000000000000000e+00, -7.000000000000000000e+00],
[-7.000000000000000000e+00, 8.000000000000000000e+00, 3.000000000000000000e+00, 4.000000000000000000e+00, -7.000000000000000000e+00, 3.000000000000000000e+00, 0.000000000000000000e+00, 8.000000000000000000e+00, 1.000000000000000000e+01, -9.000000000000000000e+00],
[6.000000000000000000e+00,-6.000000000000000000e+00, 1.000000000000000000e+00, -5.000000000000000000e+00, -8.000000000000000000e+00, 8.000000000000000000e+00, 3.000000000000000000e+00, -4.000000000000000000e+00, 7.000000000000000000e+00, -7.000000000000000000e+00],
[-0.000000000000000000e+00, -4.000000000000000000e+00, -3.000000000000000000e+00, 1.000000000000000000e+00, -7.000000000000000000e+00, -3.000000000000000000e+00, -7.000000000000000000e+00, -8.000000000000000000e+00, -3.000000000000000000e+00, -0.000000000000000000e+00],
[-7.000000000000000000e+00, -7.000000000000000000e+00, -9.000000000000000000e+00, 4.000000000000000000e+00, 7.000000000000000000e+00, -0.000000000000000000e+00, 8.000000000000000000e+00, -0.000000000000000000e+00, 9.000000000000000000e+00, 6.000000000000000000e+00],
[7.000000000000000000e+00, 2.000000000000000000e+00, 6.000000000000000000e+00, -1.000000000000000000e+01, -4.000000000000000000e+00, 9.000000000000000000e+00, -4.000000000000000000e+00, -7.000000000000000000e+00, 3.000000000000000000e+00, 4.000000000000000000e+00],
[3.000000000000000000e+00, 3.000000000000000000e+00, 2.000000000000000000e+00, 4.000000000000000000e+00, 9.000000000000000000e+00, 1.000000000000000000e+00, -9.000000000000000000e+00, 4.000000000000000000e+00, -9.000000000000000000e+00, 5.000000000000000000e+00],
[4.000000000000000000e+00, -9.000000000000000000e+00, -2.000000000000000000e+00, 1.000000000000000000e+01, 5.000000000000000000e+00, 6.000000000000000000e+00, -1.000000000000000000e+00, -5.000000000000000000e+00, 4.000000000000000000e+00, -8.000000000000000000e+00],
[5.000000000000000000e+00, 6.000000000000000000e+00, 2.000000000000000000e+00, -6.000000000000000000e+00, -7.000000000000000000e+00, 7.000000000000000000e+00, 5.000000000000000000e+00, -5.000000000000000000e+00, 7.000000000000000000e+00, 0.000000000000000000e+00],
[1.000000000000000000e+00, -3.000000000000000000e+00, -8.000000000000000000e+00, -1.000000000000000000e+00, 2.000000000000000000e+00, -8.000000000000000000e+00, 7.000000000000000000e+00, -3.000000000000000000e+00, 0.000000000000000000e+00, 6.000000000000000000e+00]])

betha = np.array([-6.000000000000000000e+00,
2.000000000000000000e+00,
5.000000000000000000e+00,
4.000000000000000000e+00,
-3.000000000000000000e+00,
-3.000000000000000000e+00,
8.000000000000000000e+00,
-6.000000000000000000e+00,
-8.000000000000000000e+00,
2.000000000000000000e+00])

C = np.array([[7.000000000000000000e+00, -1.200000000000000000e+01, 5.000000000000000000e+00, -1.100000000000000000e+01, 2.000000000000000000e+01, -1.700000000000000000e+01, 4.000000000000000000e+00, -1.700000000000000000e+01, -1.300000000000000000e+01, -2.000000000000000000e+00],
[-2.000000000000000000e+00, -1.000000000000000000e+00, -0.000000000000000000e+00, -8.000000000000000000e+00, 1.000000000000000000e+00, -0.000000000000000000e+00, -3.000000000000000000e+00, -1.100000000000000000e+01, 3.000000000000000000e+00, 9.000000000000000000e+00],
[-1.000000000000000000e+00, 1.700000000000000000e+01, 1.000000000000000000e+00, 1.000000000000000000e+01, -9.000000000000000000e+00, -1.000000000000000000e+00, -3.000000000000000000e+00, 1.300000000000000000e+01, -1.000000000000000000e+01, 1.400000000000000000e+01],
[7.000000000000000000e+00, 1.500000000000000000e+01, 1.100000000000000000e+01, 9.000000000000000000e+00, 5.000000000000000000e+00, -4.000000000000000000e+00, 8.000000000000000000e+00, -8.000000000000000000e+00, 5.000000000000000000e+00, 6.000000000000000000e+00],
[0.000000000000000000e+00, -1.100000000000000000e+01, -1.300000000000000000e+01, 4.000000000000000000e+00, -9.000000000000000000e+00, -7.000000000000000000e+00, -7.000000000000000000e+00, 1.700000000000000000e+01, 1.100000000000000000e+01, 0.000000000000000000e+00],
[3.000000000000000000e+00, 8.000000000000000000e+00, -1.300000000000000000e+01, -5.000000000000000000e+00, -1.900000000000000000e+01, 6.000000000000000000e+00, -2.300000000000000000e+01, 1.000000000000000000e+00, -1.100000000000000000e+01, -8.000000000000000000e+00],
[-3.000000000000000000e+00, 9.000000000000000000e+00, -6.000000000000000000e+00, 2.000000000000000000e+00, -9.000000000000000000e+00, -4.000000000000000000e+00, -2.000000000000000000e+00, -5.000000000000000000e+00, 1.900000000000000000e+01, 2.100000000000000000e+01],
[-6.000000000000000000e+00, 2.000000000000000000e+00, -2.400000000000000000e+01, -1.800000000000000000e+01, 8.000000000000000000e+00, -5.000000000000000000e+00, 8.000000000000000000e+00, -1.200000000000000000e+01, 1.100000000000000000e+01, -1.900000000000000000e+01],
[4.000000000000000000e+00, 5.000000000000000000e+00, -1.300000000000000000e+01, -1.000000000000000000e+00, 6.000000000000000000e+00, -6.000000000000000000e+00, 4.000000000000000000e+00, 4.000000000000000000e+00, -5.000000000000000000e+00, -1.500000000000000000e+01],
[-2.000000000000000000e+00, 7.000000000000000000e+00, 6.000000000000000000e+00, 1.000000000000000000e+01, 8.000000000000000000e+00, 1.200000000000000000e+01, -5.000000000000000000e+00, -6.000000000000000000e+00, 8.000000000000000000e+00, -6.000000000000000000e+00]])

gamma = np.array ([-9.000000000000000000e+00,
9.000000000000000000e+00,
-8.000000000000000000e+00,
-7.000000000000000000e+00,
-6.000000000000000000e+00,
-9.000000000000000000e+00,
-2.000000000000000000e+00,
0.000000000000000000e+00,
6.000000000000000000e+00,
-1.000000000000000000e+00])


#################################################
#####generacion matriz D y H y vector delta #######
#################################################

shA=np.shape(A)
shB=np.shape(B)
shC=np.shape(C)
rowTot=shA[0]+shB[0]+shC[0]
colTot=shA[1]+shB[1]+shC[1]
rowMax=np.max((shA[0],shB[0],shC[0]))
colMax=np.max((shA[1],shB[1],shA[1]))
finColB= shA[1]+shB[1]
finFilaB= shA[0]+shB[0]

D=np.zeros((rowTot,colTot))
D[0:shA[0],0:shA[1]]=A
D[shA[0]:finFilaB,shA[1]:finColB]=B
D[finFilaB:rowTot,finColB:colTot]=C

H = np.zeros((2*shA[0],3*shA[0]))
I = np.identity(shA[0])
H[0:shA[0],0:shA[1]]= I
H[0:shA[0],shA[1]:finColB]= - I

H[shA[0]:finFilaB,0:shA[1]]= I
H[shA[0]:finFilaB,finColB:colTot]= - I

#No utilizo ultima condicion ya que sino H*H.T no queda invertible
#H[finFilaB:rowTot,shA[1]:finColB]= I
#H[finFilaB:rowTot,finColB:colTot]= - I

delta = np.concatenate((alpha, betha, gamma), axis=0)


#################################################
#####SOLUCION lambda* analítico #######
#################################################

def lambdaAsterisco (w, D, H, delta):
    aux1 = np.linalg.inv(np.dot(H, H.T))
    aux2 = np.dot(D,w) - delta
    aux3 = np.dot(D.T,aux2)
    aux4 = np.dot(-H,aux3)
    return(np.dot(aux4,aux1))

#################################################
#####SOLUCION exacta analítica #######
#################################################
def solucionExacta(A, B, C, alpha, betha, gamma):
    aux1 = np.dot(A.T,alpha) + np.dot(B.T, betha) + np.dot(C.T,gamma)
    aux2 = np.dot(A.T,A) + np.dot(B.T,B) + np.dot(C.T,C)
    x = np.dot(aux1, np.linalg.inv(aux2))
    return x

#################################################
##### funciones auxiliares #######
#################################################

# Función a minimizar
def eval_f(D, w, delta):
    return (1/2)* (np.linalg.norm(np.dot(D,w) - delta))**2

# Restriccion h
def eval_h(H, w):
    return np.dot(H,w)

# Lagrangeano aumentado
def eval_L(H, w, D, delta, lambdaa, t):
    return eval_f(D, w, delta) + np.dot(lambdaa, eval_h(H, w)) + (t/2) * (np.linalg.norm(eval_h(H, w)))**2

# gradiente de Lagraneano aumentado
def gradienteLagrangeano(H, w, D, delta, lambdaa, t):
    aux1 = t * np.dot(H.T, np.dot(H,w))
    aux2 = np.dot(D.T, np.dot(D,w) - delta)
    return aux2 + np.dot(lambdaa,H) + aux1

def SolAnaliticaLagrangeano(H, D, delta, lambdaa, t):
    aux1 = np.dot(D.T,delta) - np.dot(lambdaa,H)
    aux2 = np.linalg.inv(np.dot(D.T, D) + t*np.dot(H.T, H))
    return np.dot(aux1, aux2)

# búsqueda de alpha con Armijo para descenso por gradiente
def find_Alpha_Armijo(betha, H, w, D, delta, lambdaa, t, grad):
    'inicializo variables de rango de busqueda, factor y sigma'
    s, p = 1, 0
    sigma = 0.1
    'inicializo variables necesarias para validar la desigualdad'
    func = eval_L(H, w, D, delta, lambdaa, t)
    
    while (True):
        'me muevo, calculo x en ese punto y evalúo la función'
        p = p +1
        w_i = w +((betha**p)* s * grad)
        func_betha = eval_L(H, w_i, D, delta, lambdaa, t)
        condition = sigma * (betha**p)* s *  np.dot(grad.T,grad)
        'si se da la siguiente desigualdad, encontré un punto donde la función disminuye'
        'entonces finalizo búsqueda'
        if (func_betha - func <= condition): break
    'cuando salimos del while, devuelvo el paso encontrado'
    return (betha**p)*s

# desecenso por gradiente para la búsqueda del mínimo del lagrangeano
def DescensoPorGradiente(H, w, D, delta, lambdaa, t, iterations, epsilon):   
    for i in range(iterations):  
        #calculo dirección
        direccion = - gradienteLagrangeano(H, w, D, delta, lambdaa, t)     
        #determino alpha segun armijo    
        alphaa = find_Alpha_Armijo(0.1, H, w, D, delta, lambdaa, t,  direccion)     
        #hago calculo de desenso
        w = w + (alphaa * (direccion))         
        #si la dirección es menor a epsilon, finalizo búsqueda
        if (np.linalg.norm(direccion) <= epsilon): 
            break  
    return w


def graficar(E, F, G1, G2, H):
    plt.grid(alpha=.4,linestyle='--')
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo relativo')
    plt.title('Costo Relativo en función iteraciones')
    # grafico
    plt.plot(E, 'r')
    plt.plot(F, 'g')
    plt.plot(G1, 'y')
    plt.plot(G2, 'b')
    plt.plot(H, 'm')
    plt.legend(['E', 'F', 'G1', 'G2', 'H'])
    
    plt.show()
    
def graficar2( F, G1, G2, H):
    plt.grid(alpha=.4,linestyle='--')
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo relativo')
    plt.title('Costo Relativo en función iteraciones')
    # grafico  
    plt.plot(F, 'g')
    plt.plot(G1, 'y')
    plt.plot(G2, 'b')
    plt.plot(H, 'r')
    plt.legend(['F', 'G1', 'G2', 'H'])
    
    plt.show()
    
def graficar3(x, Parte):
    plt.grid(alpha=.4,linestyle='--')
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo relativo')
    plt.title('Costo Relativo en función iteraciones Parte ' + Parte)
    # grafico
    plt.plot(x)
    plt.show()

#################################################
##### métodos a implementar #######
#################################################

  
def penalizacionCuadratica(H, D, delta, lambdaa, t0, solution):
    #iteraciones
    iteraciones_min_f = 100
    #epsilons
    epsilon_f = 1e-8
    #inicialización variables
    w = np.zeros(30)
    w_k = np.zeros(30)
    t = t0
    vector_errorRelativo = list()
    
    for k in range(iteraciones_min_f):
        t = 2 * t
        w = w_k
        #w_k = DescensoPorGradiente(H, w, D, delta, lambdaa, t, 1000, 1e-8)
        w_k = SolAnaliticaLagrangeano(H, D, delta, lambdaa, t)
        
        'guardo error relativo'
        x_k = (w_k[0:10] + w_k[10:20] + w_k[20:30]) /3
        errorRelativo = np.linalg.norm(solution - x_k)/np.linalg.norm(solution)
        vector_errorRelativo.append(errorRelativo)
        
        if ((np.linalg.norm(w_k - w) / np.linalg.norm(w_k)) < epsilon_f):
            break
    return w_k, vector_errorRelativo


def multiplicadoresLagrange(H, D, delta, lambdaa0, t, solution):
    #iteraciones
    iteraciones_min_f = 100
    iteraciones_min_lagrangeano = 1000
    #epsilons
    epsilon_f = 1e-8
    epsilon_lagrangeano = 1e-4
    #inicialización variables
    w = w_k = np.zeros(30)
    lambdaa = lambdaa0
    vector_errorRelativo = list()
    
    for k in range(iteraciones_min_f):
        lambdaa = lambdaa + t * np.dot(H, w)
        w = w_k
        w_k = DescensoPorGradiente(H, w, D, delta, lambdaa, t, iteraciones_min_lagrangeano, epsilon_lagrangeano)
        #w_k = SolAnaliticaLagrangeano(H, D, delta, lambdaa, t)
        
        'guardo error relativo'
        x_k = (w_k[0:10] + w_k[10:20] + w_k[20:30]) /3
        errorRelativo = np.linalg.norm(solution - x_k)/np.linalg.norm(solution)
        vector_errorRelativo.append(errorRelativo)
        
        if np.linalg.norm(w_k - w) / np.linalg.norm(w_k) < epsilon_f:
            break
    return w_k, vector_errorRelativo

        
def combinado(H, D, delta, lambdaa0, t0, solution):
    #iteraciones
    iteraciones_min_f = 80
    iteraciones_min_lagrangeano = 1000
    #epsilons
    epsilon_f = 1e-8
    epsilon_lagrangeano = 1e-4
    #inicialización variables
    w = w_k = np.zeros(30)
    t = t0
    lambdaa = lambdaa0
    vector_errorRelativo = list()
    
    for k in range(iteraciones_min_f):
        t = 2 * t
        lambdaa = lambdaa + t * np.dot(H, w)
        w = w_k
        w_k = DescensoPorGradiente(H, w, D, delta, lambdaa, t, iteraciones_min_lagrangeano, epsilon_lagrangeano)
        #w_k = SolAnaliticaLagrangeano(H, D, delta, lambdaa, t)
        
        'guardo error relativo'
        x_k = (w_k[0:10] + w_k[10:20] + w_k[20:30]) /3
        errorRelativo = np.linalg.norm(solution - x_k)/np.linalg.norm(solution)
        vector_errorRelativo.append(errorRelativo)
        
        if np.linalg.norm(w_k - w) / np.linalg.norm(w_k) < epsilon_f:
            break
    return w_k, vector_errorRelativo

#################################################
##### ejecución métodos #######
#################################################

'Solucion exacta'
xExacto = solucionExacta(A, B, C, alpha, betha, gamma)
print('Solución Exacta')
print(xExacto)

'Parte E'
t0 = 1/np.linalg.norm(D)
lambdaa = lambdaAsterisco (np.zeros(30), D, H, delta)
w, vector_errorRelativoE = penalizacionCuadratica(H, D, delta, lambdaa, t0, xExacto)
graficar3(vector_errorRelativoE, 'E')
print('Solución E')
x = (w[0:10] + w[10:20] + w[20:30]) /3
print(x)

'Parte F'
lambdaa = np.zeros(20)
w, vector_errorRelativoF = penalizacionCuadratica(H, D, delta, lambdaa, t0, xExacto)
graficar3(vector_errorRelativoF, 'F')
print('Solución F')
x = (w[0:10] + w[10:20] + w[20:30]) /3
print(x)


'Parte G1'
lambdaa0 = np.zeros(20)
t = (1/np.linalg.norm(D))*10
w , vector_errorRelativoG1= multiplicadoresLagrange(H, D, delta, lambdaa0, t, xExacto)
graficar3(vector_errorRelativoG1, 'G1')
print('Solución G1')
x = (w[0:10] + w[10:20] + w[20:30]) /3
print(x)


'Parte G2'
lambdaa0 = np.zeros(20)
t = (1/np.linalg.norm(D))*1000
w, vector_errorRelativoG2 = multiplicadoresLagrange(H, D, delta, lambdaa0, t, xExacto)
graficar3(vector_errorRelativoG2, 'G2')
print('Solución G2')
x = (w[0:10] + w[10:20] + w[20:30]) /3
print(x)


'Parte H'
lambdaa0 = np.zeros(20)
t0 = 1/np.linalg.norm(D)
w, vector_errorRelativoH = combinado(H, D, delta, lambdaa0, t0, xExacto)
graficar3(vector_errorRelativoH, 'H')
print('Solución H')
print(w)
x = (w[0:10] + w[10:20] + w[20:30]) /3
print(x)


graficar(vector_errorRelativoE, vector_errorRelativoF, vector_errorRelativoG1, vector_errorRelativoG2, vector_errorRelativoH)
graficar2(vector_errorRelativoF, vector_errorRelativoG1, vector_errorRelativoG2, vector_errorRelativoH)

        