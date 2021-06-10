# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

###### FUNCIONES PARA GRAFICAR #####
def graficar(x,y, Parte):
    # Add a grid
    plt.grid(alpha=.4,linestyle='--')
    # titulo y etiquetas
    plt.xlabel('eje x')
    plt.ylabel('eje y')
    plt.title('Función ' + Parte)
    # grafico
    plt.plot(x,y)
    plt.show()
    
def graficar3D(X,Y, Z, Parte):
    ax2 = plt.axes(projection="3d")
    ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax2.set_xlabel('eje x')
    ax2.set_ylabel('eje y')
    ax2.set_zlabel('eje z')
    ax2.set_title('Función ' + Parte)
    plt.show()


def graficarCurvas(x,y,z):
    plt.grid(alpha=.4,linestyle='--')
    # titulo y etiquetas
    plt.xlabel('eje x')
    plt.ylabel('eje y')
    plt.title('Curvas de nivel ')
    # grafico
    plt.contour(x, y, z)
    plt.show()

# valores de x
x = np.linspace(-1,1,100)

# Parte A
y_a = 4*x**4 - x**3 - 4*x**2 +1
graficar(x,y_a,  'parte A')

# Parte B
y_b = x**3
graficar(x,y_b, 'parte B')

# Parte C
for a in range(-2,3):
    y_c = (x-a)**2 +1
    titulo = 'parte C con a= ' + str(a)
    graficar(x,y_c,  titulo)

#Parte D
    
def eval_f(x):
    xTecho = np.array([2,0.5])
    return np.linalg.norm(x-xTecho)**2 +1

'Grafico conjunto B'    
plt.axvline(x=1)
plt.axvline(x=0)
plt.axhline(y=1)
plt.axhline(y=0)
plt.plot(0, 1, 'co')
plt.plot(0, 0, 'co')
plt.plot(1, 0, 'co')
plt.plot(1, 1, 'co')
plt.show()

'grafico función de costo'
x = np.linspace(0,1,100)
y = np.linspace(0,1,100)

z = np.ndarray((len(x),len(y)))
for i in range(len(x)):
    for j in range(len(y)):
      z[i][j] = eval_f(np.array([x[i],y[j]])) 

graficar3D(x, y, z, 'parte D')
graficarCurvas(x,y,z)