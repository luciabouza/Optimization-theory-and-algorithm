
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


def resolver(d2):
    p = cp.Variable(3)
    g1 = cp.Variable(1)
    g2 = cp.Variable(1)
    t = cp.Variable(1)
    c1 = g1
    c2 = 4 * (g2 - 40)
    cost = c1 + t
    constraints = [ [1,0,1]@p==g1, g2+[0,1,1]@p==d2, [1,-1,0]@p==10 , [-1,-1,1]@p==0 , [0,1,0]@p<= 30 ,[0,-1,0]@p<= 30, g1>=0, g2>=0, c2<=t , 0<=t]
    prob = cp.Problem(cp.Minimize(cost),constraints)
    prob.solve()
    return prob.value, g1, g2, p, constraints

g1, g2, p2, l = [], [], [], []


for d2 in range(201):
    value, g1i, g2i, p, constraints = resolver(d2)
    g1.append(g1i.value)
    g2.append(g2i.value)
    p2.append(p[1].value)
    l.append(constraints[2].dual_value)

plt.grid()
plt.xlabel('d2')
plt.plot(g1)
plt.plot(g2)
plt.plot(p2)
plt.plot(l)
plt.legend(['g1', 'g2', 'p2', 'lambda'])
plt.show()
