import numpy as np
from sympy import symbols

from numerical_analysis.supplement import DLU_decomposition, is_symmetric_matrix, compute_V, LU_decomposition,\
    QR_decomposition


# 1.乘幂法
def power_method(A, m, p=0):
    n = A.shape[0]
    lambd = np.zeros((1, n))
    feature_vector = np.zeros((n, n))
    A = A-p*np.identity(n)
    for i in range(n):
        u0 = np.ones((n, 1))
        v_l = np.dot(A, u0)
        u_l = v_l/max(v_l)
        v_m = np.dot(A, u_l)
        u_m = v_m/max(v_m)
        for j in range(m-1):
            v_l = v_m
            u_l = u_m
            v_m = np.dot(A, u_l)
            u_m = v_m / max(v_m)
        for k in range(n):
            feature_vector[k, i] = u_l[k, 0]
        index = np.where(abs(v_l)==max(abs(v_l)))
        lambd[0, i] = v_l[index[0]]
        A = A-lambd[0, i]*np.dot(u_l, u_l.T)/np.dot(u_l.T, u_l)
    return lambd, feature_vector

# 2.Jacobi方法
def jacobi_method(A, threshold=1e-6):
    Cod = is_symmetric_matrix(A)
    if Cod == False:
        print('This method is not applicable.')
        return
    m = A.shape[0]
    lambd = np.zeros((1, m))
    T = A.copy()
    Q = np.identity(m)
    D, L, U = DLU_decomposition(T)
    max_num = np.max(abs(U))
    while max_num >= threshold:
        index = np.where(abs(U) == max_num)
        p = index[0][0]
        q = index[1][0]
        b = (T[q, q]-T[p, p])/2/T[p, q]
        if b >= 0:
            d = 1/(b+np.sqrt(1+b**2))
        else:
            d = 1/(b-np.sqrt(1+b**2))
        c = 1/np.sqrt(1+d**2)
        S = np.identity(m)
        S[p, p] = c
        S[q, q] = c
        S[p, q] = d*c
        S[q, p] = -d*c
        T = np.dot(np.dot(S.T, T), S)
        Q = np.dot(Q, S)
        D, L, U = DLU_decomposition(T)
        max_num = np.max(abs(U))
    for i in range(m):
        lambd[0, i] = T[i, i]
    feature_vector = Q
    return lambd, feature_vector

# 3.Givens方法
def givens_method(A):
    Cod = is_symmetric_matrix(A)
    if Cod == False:
        print('This method is not applicable.')
        return
    m = A.shape[0]
    T = A.copy()
    for i in range(m-2):
        for j in range(i+2, m):
            p = i
            q = i+1
            r = j
            S = np.identity(m)
            S[q, q] = T[p, q]/(np.sqrt(T[p, q]**2+T[p, r]**2))
            S[r, r] = S[q, q]
            S[q, r] = -T[p, r]/(np.sqrt(T[p, q]**2+T[p, r]**2))
            S[r, q] = -S[q, r]
            T = np.dot(np.dot(S.T, T), S)
    return T

# 4.Householder方法
def householder_method(A):
    Cod = is_symmetric_matrix(A)
    if Cod == False:
        print('This method is not applicable.')
        return
    m = A.shape[0]
    T = A.copy()
    for i in range(m-2):
        a = T[i+1:m, i].reshape(m-i-1, 1)
        sigma = np.sign(A[i+1, i])*np.linalg.norm(a)
        e = np.zeros((m-i-1, 1))
        e[0, 0] = 1
        v = (a+sigma*e)/np.linalg.norm(a+sigma*e)
        P = np.identity(m-i-1)-2*np.dot(v, v.T)
        U = np.identity(m)
        U[i+1:m, i+1:m] = P
        T = np.dot(np.dot(U, T), U)
    return T

# 5.对称三对角矩阵的特征值计算
def sturm_method(A, point, epsilon=1e-6):
    x = symbols('x')
    m = A.shape[0]
    f = []
    f.append(1*x**0)
    f.append(x-A[0, 0])
    for i in range(1, m):
        f.append((x-A[i, i])*f[i]-A[i, i-1]**2*f[i-1])
    f.reverse()
    lambd = np.zeros((1, m))
    for i in range(m):
        a = point[i]
        b = point[i+1]
        while (b-a)/2 >= epsilon:
            mid = (a + b) / 2
            if compute_V(f, a) == compute_V(f, mid):
                a = mid
            else:
                b = mid
        lambd[0, i] = mid
    return lambd

# 6.LR算法
def LR_method(A, iteration=10):
    m = A.shape[0]
    T = A.copy()
    lambd = np.zeros((1, m))
    for i in range(iteration):
        F, G = LU_decomposition(T)
        T = np.dot(G, F)
    for i in range(m):
        lambd[0, i] = T[i, i]
    return lambd

# 7.QR算法
def QR_method(A, iteration=10):
    m = A.shape[0]
    T = A.copy()
    lambd = np.zeros((1, m))
    for i in range(iteration):
        F, G = QR_decomposition(T)
        T = np.dot(G, F)
    for i in range(m):
        lambd[0, i] = T[i, i]
    return lambd