import cvxpy as cp
import numpy as np
from itertools import product

Q0 = np.array ([ 
1.0000000e+00,
2.0000000e+00,
3.0000000e+00,
4.0000000e+00,
5.0000000e+00,
6.0000000e+00,
7.0000000e+00,
8.0000000e+00,
9.0000000e+00,
1.0000000e+01])

R0 = 1.0000000e+02
      
P0= np.array([[
5.0000000e+00,   1.0000000e+00,   5.0000000e-01,   5.0000000e-01,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00],
[1.0000000e+00,   5.0000000e+00,   1.0000000e+00,   5.0000000e-01,   5.0000000e-01,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00],
[5.0000000e-01,   1.0000000e+00,   5.0000000e+00,   1.0000000e+00,   5.0000000e-01,   5.0000000e-01,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00],
[5.0000000e-01,   5.0000000e-01,   1.0000000e+00,   5.0000000e+00,   1.0000000e+00,   5.0000000e-01,   5.0000000e-01,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00],
[0.0000000e+00,   5.0000000e-01,   5.0000000e-01,   1.0000000e+00,   5.0000000e+00,   1.0000000e+00,   5.0000000e-01,   5.0000000e-01,   0.0000000e+00,   0.0000000e+00],
[0.0000000e+00,   0.0000000e+00,   5.0000000e-01,   5.0000000e-01,   1.0000000e+00,   5.0000000e+00,   1.0000000e+00,   5.0000000e-01,   5.0000000e-01,   0.0000000e+00],
[0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   5.0000000e-01,   5.0000000e-01,   1.0000000e+00,   5.0000000e+00,   1.0000000e+00,   5.0000000e-01,   5.0000000e-01],
[0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   5.0000000e-01,   5.0000000e-01,   1.0000000e+00,   5.0000000e+00,   1.0000000e+00,   5.0000000e-01],
[0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   5.0000000e-01,   5.0000000e-01,   1.0000000e+00,   5.0000000e+00,   1.0000000e+00],
[0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   5.0000000e-01,   5.0000000e-01,   1.0000000e+00,   5.0000000e+00]])


def resolverDB(P0, Q0, R0, n):
    t = cp.Variable(1)
    lambdaa = cp.Variable(n)
    P = cp.Variable((n,n))
    Q = cp.Variable((n,1))
    X = cp.Variable((n+1,n+1), PSD=True)
    
    Q0_vector = Q0.reshape(Q.shape)
    
    P = P0 + cp.diag(lambdaa)
    Q = Q0_vector
    R = R0 - sum(lambdaa)
    
    func = t
    restricciones = [ X>=0, X == cp.bmat([[P, Q], [Q.T, cp.diag(R - t)]])]
    
    prob = cp.Problem(cp.Maximize(func),restricciones)
    prob.solve()
    return prob.value



def resolverPA(P0, Q0, R0, n):
    x = cp.Variable((n,1))   
      
    func = cp.quad_form(x, P0) + 2 * Q0.T @ x + R0 
    restricciones = [-1<=x, x<=1]
    
    prob = cp.Problem(cp.Minimize(func),restricciones)
    prob.solve()
	    
    xPA = np.sign(x.value)
    return xPA
	
   
def resolverPBFuerzaBruta(P0, Q0, R0, n):
    vectors = list(product([-1,1], repeat= n))
    vaux = np.array(vectors[0])
    vmin = vaux 
    minimo = np.dot(np.dot(vaux.T,P0),vaux) + 2*np.dot(Q0.T,vaux) + R0
    for v in vectors:
        varr= np.array(v)
        fv = np.dot(np.dot(varr.T,P0),varr) + 2*np.dot(Q0.T,varr) + R0
        if fv < minimo: 
            minimo = fv
            vmin = varr
    return minimo, vmin
    

# resolucion de problemas#  

fPB, xmin = resolverPBFuerzaBruta(P0, Q0, R0, 10)
print('valores fuerza bruta PB x= %s , f* = %s' % (xmin, fPB))

dDB = resolverDB(P0, Q0, R0, 10)
print('valor problema dual  d* = %s' % (dDB))

xPA = resolverPA(P0, Q0, R0, 10)
vxPA = np.dot(np.dot(xPA.T,P0),xPA) + 2*np.dot(Q0.T,xPA) + R0
print('valor problema PA  PA* = %s' % (vxPA))
