import numpy as np
from sympy import *
from numerical_analysis.supplement import Gauss_elimination, SMF_no_zero, is_SPDM, is_TM, DLU_decomposition, \
    LU_decomposition
from numerical_analysis.supplement import Gauss_Jordan_elimination

# 1.高斯消元法
def guass(A, b):
    m = A.shape[0]
    x = np.zeros((m, 1))
    B = Gauss_elimination(np.hstack((A, b)))
    A_proc = B[:, 0:m]
    b_proc = B[:, m].reshape(m,1)
    for i in reversed(range(m)):
        x[i, 0] = (b_proc[i, 0]-np.sum(np.dot(A_proc[i, :], x)))/A_proc[i, i]
    return x

# 2.高斯若当消元法
def guass_jordan(A, b):
    m = A.shape[0]
    x = np.zeros((m, 1))
    B = Gauss_Jordan_elimination(np.hstack((A, b)))
    A_proc = B[:, 0:m]
    b_proc = B[:, m].reshape(m, 1)
    for i in range(m):
        x[i, 0] = b_proc[i, 0]/A_proc[i, i]
    return x

# 3.Doolittle分解方法
def doolittle_decomposition(A, b):
    Cod = SMF_no_zero(A)
    if Cod == False:
        print('This method is not applicable.')
        return
    m = A.shape[0]
    L, U = LU_decomposition(A)
    y = np.zeros((m, 1))
    for i in range(m):
        y[i, 0] = b[i, 0]-np.sum(np.dot(L[i, :], y))
    x = np.zeros((m, 1))
    for i in reversed(range(m)):
        x[i, 0] = (y[i, 0]-np.sum(np.dot(U[i, :], x)))/U[i, i]
    return x

# 4.Crout分解方法
def crout_decomposition(A, b):
    Cod = SMF_no_zero(A)
    if Cod == False:
        print('This method is not applicable.')
        return
    m = A.shape[0]
    L = np.zeros((m, m))
    U = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            L[j, i] = A[j, i]-np.sum(np.dot(L[j, :], U[:, i]))
            if j > i:
                U[i, j] = (A[i, j]-np.sum(np.dot(L[i, :], U[:, j])))/L[i, i]
            U[i, i] = 1
    y = np.zeros((m, 1))
    for i in range(m):
        y[i, 0] = (b[i, 0] - np.sum(np.dot(L[i, :], y)))/L[i,i]
    x = np.zeros((m, 1))
    for i in reversed(range(m)):
        x[i, 0] = y[i, 0] - np.sum(np.dot(U[i, :], x))
    return x

# 5.Cholesky分解方法
def cholesky_decomposition(A, b):
    Cod = is_SPDM(A)
    if Cod == False:
        print('This method is not applicable.')
        return
    m = A.shape[0]
    L = np.zeros((m, m))
    for i in range(m):
        temp = 0
        for k in range(i):
            temp += L[i, k]*L[i, k]
        L[i, i] = np.sqrt(A[i, i]-temp)
        for j in range(i+1, m):
            temp = 0
            for k in range(i):
                temp += L[j, k]*L[i, k]
            L[j, i] = (A[j, i]-temp)/L[i, i]
    y = np.zeros((m, 1))
    for i in range(m):
        temp = 0
        for j in range(i):
            temp += L[i, j]*y[j, 0]
        y[i, 0] = (b[i, 0]-temp)/L[i, i]
    x = np.zeros((m, 1))
    for i in reversed(range(m)):
        temp = 0
        for j in range(i+1, m):
            temp += L[j, i]*x[j, 0]
        x[i, 0] = (y[i, 0]-temp)/L[i, i]
    return x


# 6.改进的Cholesky分解方法
def improved_cholesky_decomposition(A, b):
    Cod = is_SPDM(A)
    if Cod == False:
        print('This method is not applicable.')
        return
    m = A.shape[0]
    L = np.zeros((m, m))
    d = np.zeros((m, 1))
    for i in range(m):
        d[i, 0] = A[i, i]-np.sum(np.multiply(np.multiply(L[i, :], L[i, :]), d.T))
        for j in range(i+1, m):
            L[j, i] = (A[j, i]-np.sum(np.multiply(np.multiply(L[j, :], d.T), L[i, :])))/d[i, 0]
        L[i, i] = 1
    y = np.zeros((m, 1))
    for i in range(m):
        y[i, 0] = b[i, 0]-np.dot(L[i, :], y)
    x = np.zeros((m, 1))
    for i in reversed(range(m)):
        x[i, 0] = y[i, 0]/d[i, 0]-np.dot(L[:, i], x)
    return x

# 7.追赶法
def catch_up(A, b):
    Cod = is_TM(A)
    if Cod == False:
        print('This method is not applicable.')
        return
    m = A.shape[0]
    a_prem = np.zeros((m-1, 1))
    b_prem = np.zeros((m, 1))
    c_prem = np.zeros((m-1, 1))
    d_prem = np.zeros((m, 1))
    for i in range(m-1):
        a_prem[i, 0] = A[i+1, i]
        b_prem[i, 0] = A[i, i]
        c_prem[i, 0] = A[i, i+1]
        d_prem[i, 0] = b[i, 0]
    b_prem[m-1, 0] = A[m-1, m-1]
    d_prem[m-1, 0] = b[m-1, 0]
    bet = np.zeros((m-1, 1))
    bet[0, 0] = c_prem[0, 0]/b_prem[0, 0]
    for i in range(1, m-1):
        bet[i, 0] = c_prem[i, 0]/(b_prem[i, 0]-a_prem[i-1, 0]*bet[i-1, 0])
    y = np.zeros((m, 1))
    y[0, 0] = d_prem[0, 0]/b_prem[0, 0]
    for i in range(1, m):
        y[i, 0] = (d_prem[i, 0]-a_prem[i-1, 0]*y[i-1, 0])/(b_prem[i, 0]-a_prem[i-1, 0]*bet[i-1, 0])
    x = np.zeros((m, 1))
    x[m-1, 0] = y[m-1, 0]
    for i in reversed(range(m-1)):
        x[i, 0] = y[i, 0]-bet[i, 0]*x[i+1, 0]
    return x

# 8.病态问题检验
def Cond(A):
    A_inv = np.linalg.inv(A)
    A_norm = np.linalg.norm(A, ord=1)
    A_inv_norm = np.linalg.norm(A_inv, ord=1)
    cond = A_norm*A_inv_norm
    return cond

# 9.Jacobi迭代
def jacobi_iterative(A, b, x_initial, iteration=10, accuracy=0.0000001):
    m = A.shape[0]
    D, L, U = DLU_decomposition(A)
    if np.linalg.det(D) == 0:
        print('This method is not applicable.')
        return
    Im = np.identity(m)
    B_j = Im-np.dot(np.linalg.inv(D),A)
    x = x_initial.copy()
    i = 0
    while i < iteration:
        r = max(abs(b - np.dot(A, x)))
        if r < accuracy:
            break
        else:
            x = np.dot(B_j, x)+np.dot(np.linalg.inv(D), b)
        i += 1
    return x

# 10.Gauss_Seidel迭代
def gauss_seidel_iterative(A, b, x_initial, iteration=10, accuracy=0.0000001):
    m = A.shape[0]
    D, L, U = DLU_decomposition(A)
    if np.linalg.det(D + L) == 0:
        print('This method is not applicable.')
        return
    B_g = -np.dot(np.linalg.inv(D+L), U)
    x = x_initial.copy()
    i = 0
    while i < iteration:
        r = max(abs(b - np.dot(A, x)))
        if r < accuracy:
            break
        else:
            x = np.dot(B_g, x) + np.dot(np.linalg.inv(D+L), b)
        i += 1
    return x

# 11.超松弛迭代法
def SOR_iterative(A, b, x_initial, w=1, iteration=10, accuracy=0.0000001):
    m = A.shape[0]
    D, L, U = DLU_decomposition(A)
    if np.linalg.det(D + w*L) == 0:
        print('This method is not applicable.')
        return
    B_w = np.dot(np.linalg.inv(D+w*L), (1-w)*D-w*U)
    x = x_initial.copy()
    i = 0
    while i < iteration:
        r = max(abs(b - np.dot(A, x)))
        if r < accuracy:
            break
        else:
            x = np.dot(B_w, x) + w*np.dot(np.linalg.inv(D+w*L), b)
        i += 1
    return x

# 12.最速下降法
def steepest_descent(A, b, x_initial, iteration=30, accuracy=0.0000001):
    Cod = is_SPDM(A)
    if Cod == False:
        print('This method is not applicable.')
        return
    x = x_initial
    r = b - np.dot(A, x)
    i = 0
    while i < iteration:
        if max(abs(r)) < accuracy:
            break
        else:
            alpha = np.dot(r.T, r)/np.dot(np.dot(A, r).T, r)
            x = x+alpha*r
            r = b - np.dot(A, x)
        i += 1
    return x

# 13.共轭梯度法
def conjugate_gradient(A, b, x_initial, epsilon=0.000001):
    Cod = is_SPDM(A)
    if Cod == False:
        print('This method is not applicable.')
        return
    x = x_initial
    r = b - np.dot(A, x)
    p = r
    alpha = np.dot(r.T, r)/np.dot(p.T, np.dot(A, p))
    x = x+alpha*p
    r_new = r - alpha*np.dot(A, p)
    while np.dot(r_new.T, r_new)/np.dot(b.T, b) >= epsilon:
        beta = np.dot(r_new.T, r_new)/np.dot(r.T, r)
        p = r_new+beta*p
        r = r_new.copy()
        alpha = np.dot(r.T, r) / np.dot(p.T, np.dot(A, p))
        x = x + alpha * p
        r_new = r - alpha * np.dot(A, p)
    return x


