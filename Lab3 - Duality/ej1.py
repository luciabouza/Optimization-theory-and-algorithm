import numpy as np
import matplotlib.pyplot as plt



def f1(x):
    return x

def f4(x):
    return x**3

def g1(x):
    return x**2 -1 

def g2(x):
    return x**2

def g3(x):
    return abs(x)

def g4(x):
    return -x+1

# debe tener pendiente -mu* y pasar por el punto (0,f*)
# esta es la recta que pasa minimiza el lagrangeano 
def recta_mu1(t):
    return -0.5*t -1

def recta_mu3(t, mu):
    return -mu*t

def conjuntoA(x, f, g):
    # valores de z y t para generar conjunto A. 
    # agregaremos solo los que cumplan la condicion que exista x/ g(x)<=t y f(x)<=z
    z1 = []
    t1 = []
    
    zAux = np.linspace(-5,5,100)
    tAux = np.linspace(-5,5,100)
    for xi in x:
        for zi in zAux:
            for ti in tAux:
                if (f(xi)<=zi and g(xi)<=ti):
                    z1.append(zi)
                    t1.append(ti)
    return (t1, z1)


x = np.linspace(-2,2,400)

############## F1 #######################
z = f1(x)
t = g1(x)
recta_mu = recta_mu1(t)
t1, z1 = conjuntoA(x, f1, g1)

plt.grid(alpha=.4,linestyle='--')
plt.plot(t,z) # Conjunto S
plt.plot(t1,z1, alpha=0.2) # Conjunto A
plt.plot(t,recta_mu) # rectamu
plt.show()


############# F2 #######################

z = f1(x)
t = g2(x)

t1, z1 = conjuntoA(x, f1, g2)

plt.grid(alpha=.4,linestyle='--')
plt.plot(t,z) # Conjunto S
plt.plot(t1,z1, alpha=0.2) # Conjunto A
plt.axvline(x=0) #recta mu
plt.show()


############# F3 #######################

z = f1(x)
t = g3(x)
t1, z1 = conjuntoA(x, f1, g3)

for mu in range(4):
    x = np.linspace(0,2,10)
    recta_mu = recta_mu3(x, mu+2)
    plt.plot(x,recta_mu) # rectamu

plt.grid(alpha=.4,linestyle='--')
plt.plot(t,z) # Conjunto S
plt.plot(t1,z1, alpha=0.2) # Conjunto A

plt.show()

############# F4 #######################

x = np.linspace(-2,2,400)
z = f4(x)
t = g4(x)
t1, z1 = conjuntoA(x, f4, g4)

plt.grid(alpha=.4,linestyle='--')
plt.plot(t,z) # Conjunto S
plt.plot(t1,z1, alpha=0.2) # Conjunto A

plt.show()