# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def func(x,y):
    return (-np.log(y**2 - x**2)) 

######################################################
#### GRAFICO DEL CONJUNTO FACTIBLE ###################   
######################################################  

x = np.linspace(0, 1, 100)

'graficas de los límites de las restricciones. x>=0 está dado por los puntos elegidos para la variable x'
f = np.sqrt(1 -x**2)
g = np.full(100, 0.5)
h = 2*x

'grafico las funciones'
plt.plot(x, f, label = "f = np.sqrt(1 -x**2)")
plt.plot(x, g, label = "g = 0.5")
plt.plot(x, h, label = "h = 2*x")
plt.axvline(x=0, label = "x=0")
ax = plt.axes()
ax.set_xlabel('x')
ax.set_ylabel('y')

'marco en rojo las intersecciones de los limites del conjunto factible'
idx = np.argwhere(np.diff(np.sign(f - h))).flatten()
plt.plot(x[idx], f[idx], 'ro')
idx = np.argwhere(np.diff(np.sign(g - h))).flatten()
plt.plot(x[idx], g[idx], 'ro')
plt.plot(0, g[0], 'ro')
plt.plot(0, f[0], 'ro')

plt.legend()
plt.show()


######################################################
#### GRAFICO DE LA FUNCION Y CURVAS DE NIVEL #########   
######################################################  
x = np.linspace(0,0.5,1000)
y = np.linspace(0.5,1,1000)
X, Y = np.meshgrid(x, y)
Z = func(X, Y)

fig = plt.figure()
ax2 = plt.axes(projection="3d")
ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f')
ax2.set_title('Función')
plt.show()

ax2 = plt.axes(projection='3d')
ax2.contour(X, Y, Z, levels = 30, extend3d=True)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f')
ax2.set_title('Curvas de nivel')
plt.show()


