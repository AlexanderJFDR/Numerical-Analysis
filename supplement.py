import numpy as np
from matplotlib import pyplot as plt
from sympy import *


def compute_Jocab(f, p, n=8):
    m = len(f)
    A = np.mat(np.zeros((m, m)))
    x = symbols('x1:%d'%(m+1))
    for i in range(m):
        for j in range(m):
            A[i, j] = diff(f[i], x[j]).subs(dict(zip(x, p))).evalf(n)
    return A

def Gauss_elimination(B):
    A = B.copy()
    m = A.shape[0]
    n = A.shape[1]
    for i in range(m-1):
        index = np.argmax(abs(A[i:m, i]))+i
        A[[i, index], :] = A[[index, i], :]
        for j in range(i+1, m): #行
            temp = A[j, i]/A[i, i]
            for k in range(i, n): #列
                A[j, k] = A[j, k] - temp*A[i, k]
    return A

def Gauss_Jordan_elimination(B):
    A = B.copy()
    m = A.shape[0]
    n = A.shape[1]
    for i in range(m):
        index = np.argmax(abs(A[i:m, i])) + i
        A[[i, index], :] = A[[index, i], :]
        for j in range(m):  # 行
            if j != i:
                temp = A[j, i] / A[i, i]
                for k in range(i, n):  # 列
                    A[j, k] = A[j, k] - temp * A[i, k]
    return A

def SMF_no_zero(A):  #顺序主子式均不为零
    m = A.shape[0]
    for i in range(m):
        det = np.linalg.det(A[0:i+1, 0:i+1])
        if det == 0:
            return False
    return True

def is_SPDM(A):  #对称正定矩阵
    m = A.shape[0]
    B = A-A.T
    if (abs(B) > 0).any():
        return False
    for i in range(m):
        det = np.linalg.det(A[0:i + 1, 0:i + 1])
        if det <= 0:
            return False
    return True

def is_TM(A):  #三对角阵且对角占优
    m = A.shape[0]
    for i in range(m):
        for j in range(m):
            if A[i, j] != 0 and abs(i-j) > 1:
                return False
            if i==0 and abs(A[i, i])<=A[i, i+1]:
                return False
            elif i==m-1 and abs(A[i, i]<=A[i, i-1]):
                return False
            elif i>0 and i<m-1 and abs(A[i, i])<(abs(A[i, i-1])+abs(A[i, i+1])):
                return False
    return True

def DLU_decomposition(A):
    m = A.shape[0]
    D = np.zeros((m, m))
    L = np.zeros((m, m))
    U = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i > j:
                L[i, j] = A[i, j]
            elif i == j:
                D[i, j] = A[i, j]
            elif i < j:
                U[i, j] = A[i, j]
    return D, L, U

def draw_picture(a, b, y, n=100):
    x = symbols('x')
    t = np.linspace(min(a), max(a), n)
    y_t = np.linspace(0, 0, n)
    for i in range(n):
        y_t[i] = y.subs({x: t[i]}).evalf(6)
    plt.plot(t, y_t, 'r', linewidth=2)
    plt.scatter(a, b, alpha=1)
    plt.show()

def draw_piecewised_picture(a, b, y, n=10):
    m = len(y)
    x = symbols('x')
    y_t = np.linspace(0, 0, n)
    for i in range(m):
        t = np.linspace(a[i], a[i+1], n)
        for j in range(n):
            y_t[j] = y[i].subs({x: t[j]}).evalf(6)
        plt.plot(t, y_t, 'r', linewidth=2)
    plt.scatter(a, b, alpha=1)
    plt.show()

def difference_quotient(a, b):
    m = len(a)
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i+1):
            if j == 0:
                D[i, j] = b[i]
            else:
                D[i, j] = (D[j-1, j-1]-D[i, j-1])/(a[j-1]-a[i])
    return D

def creat_p(a, r):
    x = symbols('x')
    y = 1
    for i in range(r):
        y = y*(x-a[i])
    return y

def creat_l(a, j, r):
    x = symbols('x')
    l = creat_p(a, r)/(x-a[j])/diff(creat_p(a, r),x).subs({x:a[j]})
    return l

