import numpy as np
from matplotlib import pyplot as plt
from sympy import *
import csv
import pandas as pd

def differential_equations_runge_kutte4(f, C, initial_point, step=0.01, is_draw=True):
    m = len(f)
    x = symbols('x1:%d' % (m+1))
    y = symbols('y1:%d' % (m+1))
    [a, b] = C
    num_point = int(np.floor((b - a) / step+1e-9))
    h = (b-a)/num_point
    X = np.zeros((m, num_point))
    Y = np.zeros((m, num_point))
    X[:, 0] = initial_point[:, 0]
    Y[:, 0] = initial_point[:, 1]
    for i in range(1, num_point):
        X[:, i] = X[:, i-1]+h
        K = np.zeros((m, 4))
        for j in range(m):
            K[j, 0] = f[j].subs(dict(zip([*x, *y], [*X[:,i-1], *Y[:,i-1]])))
        for j in range(m):
            K[j, 1] = f[j].subs(dict(zip([*x, *y], [*(X[:,i-1]+h/2), *(Y[:,i-1]+h/2*K[:,0])])))
        for j in range(m):
            K[j, 2] = f[j].subs(dict(zip([*x, *y], [*(X[:,i-1]+h/2), *(Y[:,i-1]+h/2*K[:,1])])))
        for j in range(m):
            K[j, 3] = f[j].subs(dict(zip([*x, *y], [*(X[:,i-1]+h/2), *(Y[:,i-1]+h/2*K[:,2])])))
        Y[:, i] = Y[:, i-1]+h/6*(K[:,0]+2*K[:,1]+2*K[:,2]+K[:,3])
    if is_draw == True:
        plt.figure()
        plt.plot(X[1, :], Y[2, :], 'r', linewidth=2)
        plt.figure()
        plt.plot(X[1, :], Y[4, :], 'r', linewidth=2)
        plt.show()
    return X, Y

def main():
    x1 = symbols('x1')
    y1 = symbols('y1')
    y2 = symbols('y2')
    y3 = symbols('y3')
    y4 = symbols('y4')
    y5 = symbols('y5')
    y6 = symbols('y6')
    f = []
    f.append(y2)
    f.append(4.3793*y3+0.2003*cos(2*pi/1.7152*x1)-0.0776*y4-120.6024*y5-0.5041*y6)
    f.append(y4)
    f.append(0.4977*y3+0.159*cos(2*pi/1.7152*x1)-0.0615*y4-29.6097*y5-0.1238*y6)
    f.append(y5)
    f.append(25.5585*y3+0.2092*cos(2*pi/1.7152*x1)-0.081*y4-591.749*y5-2.474*y6)
    initial_point = np.array([[0, 0],[0, 0],[0, 0],[0, 0],[0, 0],[0, 0]])
    X, Y = differential_equations_runge_kutte4(f, [0, 56], initial_point, step=0.01, is_draw=True)

    print(X)
    print(Y)

if __name__ == '__main__':
    main()