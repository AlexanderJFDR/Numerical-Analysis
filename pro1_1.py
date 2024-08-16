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
        plt.plot(X[1, :], Y[2, :]-Y[0, :]-1.002, 'r', linewidth=2)
        plt.show()
        plt.figure()
        plt.plot(X[1, :], Y[0, :], 'r', linewidth=2)
        plt.show()
        plt.figure()
        plt.plot(X[1, :], Y[2, :], 'r', linewidth=2)
        plt.show()
    return X, Y

def main():
    x1 = symbols('x1')
    y1 = symbols('y1')
    y2 = symbols('y2')
    y3 = symbols('y3')
    y4 = symbols('y4')
    f = []
    f.append(y2)
    f.append((-10656.3616*y2+10000*y4-111557*y1+80000*y3+6250*cos(2*pi/1.4005*x1)-80156.6)/6201.535)
    f.append(y4)
    f.append((10000*y2-10000*y4+80000*y1-80000*y3+56313.2)/2433)
    initial_point = np.array([[0, 0],[0, 0],[0, 1.002],[0, 0]])
    X, Y = differential_equations_runge_kutte4(f, [0, 56], initial_point, step=0.01, is_draw=True)
    list=[]
    y=zeros(1,280)
    for i in range(4):
        for j in range(0,5600,20):
            y[j]=Y[i,j]
        list.append(y)
    with open('data.csv', 'w', newline="") as f0:
        csv_writer0 = csv.writer(f0)
        for i in list:
            csv_writer0.writerow(i)

    print(X)
    print(Y)

if __name__ == '__main__':
    main()