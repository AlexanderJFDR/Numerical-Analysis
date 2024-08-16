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
        #index = i
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

def LU_decomposition(A):
    m = A.shape[0]
    L = np.zeros((m, m))
    U = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            U[i, j] = A[i, j] - np.sum(L[i, :] * U[:, j])
            if j == i:
                L[j, i] = 1.0
            else:
                L[j, i] = (A[j, i] - np.sum(L[j, :] * U[:, i])) / U[i, i]
    return L, U

def QR_decomposition(A):
    m = A.shape[0]
    B = np.zeros((m, m))
    U = np.zeros((m, m))
    R = np.zeros((m, m))
    for i in range(m):
        b = A[:, i]
        for j in range(i):
            b = b-B[:, j]*(np.dot(A[:, i].T, B[:, j])/np.dot(B[:, j].T, B[:, j]))
        B[:, i] = b
    for i in range(m):
        U[:, i] = B[:, i]/np.linalg.norm(B[:, i])
        R[i, i] = np.linalg.norm(B[:, i])
        for j in range(i):
            R[j, i] = np.dot(A[:, i].T, B[:, j])/np.linalg.norm(B[:, j])
    return U, R

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

# 整体插值
def draw_picture(a, b, y, n=100):
    x = symbols('x')
    t = np.linspace(min(a), max(a), n)
    y_t = np.linspace(0, 0, n)
    for i in range(n):
        y_t[i] = y.subs({x: t[i]}).evalf(6)
    plt.plot(t, y_t, 'r', linewidth=2)
    plt.scatter(a, b, alpha=1)
    plt.show()

# 分段插值
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

# 连续函数逼近
def draw_approximation_curve(g, y, a, b, n=100):
    x = symbols('x')
    t = np.linspace(a, b, n)
    y_t = np.linspace(0, 0, n)
    g_t = np.linspace(0, 0, n)
    for i in range(n):
        y_t[i] = y.subs({x: t[i]}).evalf(6)
        g_t[i] = g.subs({x: t[i]}).evalf(6)
    plt.plot(t, g_t, 'b', linewidth=2, label='original curve')
    plt.plot(t, y_t, 'r', linewidth=2, linestyle='--', label='approximate curve')
    plt.show()

# 离散数据逼近
def draw_polynomial_fitting_curve(a, b, y, n=100):
    x = symbols('x')
    t = np.linspace(min(a), max(a), n)
    y_t = np.linspace(0, 0, n)
    for i in range(n):
        y_t[i] = y.subs({x: t[i]}).evalf(6)
    plt.plot(t, y_t, 'r', linewidth=2)
    plt.scatter(a, b, alpha=1)
    plt.show()

def creat_Legendre(m):
    x = symbols('x')
    f = []
    f.append(1)
    for i in range(1, m):
        f.append(1/(2**i*factorial(i))*diff((x**2-1)**i, x, i))
    return f

def creat_Chebyshev1(m):
    x = symbols('x')
    f = []
    for i in range(m):
        f.append(cos(i*acos(x)))
    return f

def creat_Chebyshev2(m):
    x = symbols('x')
    f = []
    for i in range(m):
        f.append(sin((i+1)*acos(x))/sqrt(1-x**2))
    return f

def creat_Laguerre(m):
    x = symbols('x')
    f = []
    for i in range(m):
        f.append(exp(x)*diff(x**i*exp(-x), x, i))
    return f

def creat_Hermite(m):
    x = symbols('x')
    f = []
    for i in range(m):
        f.append((-1)**i*exp(x**2)*diff(exp(-x**2), x, i))
    return f

# 画频域图
def draw_frequency_domain_diagram(a_0, a, b):
    m = len(a)
    p = np.linspace(0, 0, m+1)
    p[0] = a_0**2/2
    p[1:] = a**2+b**2
    plt.bar(range(m+1), p)
    plt.show()

def creat_Newton_Cotes_coefficient():
    W = np.array([[1/2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1/6, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1/8, 1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0],
         [1/90, 7, 32, 12, 32, 7, 0, 0, 0, 0, 0, 0],
         [1/288, 19, 75, 50, 50, 75, 19, 0, 0, 0, 0, 0],
         [1/840, 41, 216, 27, 272, 27, 216, 41, 0, 0, 0, 0],
         [1/17280, 751, 3577, 1323, 2989, 2989, 1323, 3577, 751, 0, 0, 0],
         [1/28350, 989, 5888, -928, 10496, -4540, 10496, -928, 5888, 989, 0, 0],
         [1/89600, 2857, 15741, 1080, 19344, 5778, 5778, 19344, 1080, 15741, 2857, 0],
         [1/598752, 16067, 106300, -48525, 272400, -260550, 427368, -260550, 272400, -48525, 106300, 16067]])
    return W

def creat_Gauss_coefficient(option):
    W = []
    if option == 'Legendre':
        W.append([[0],[2]])
        W.append([[-0.5773502692,0.5773502692],[1,1]])
        W.append([[-0.7745966692,0,0.7745966692],[5/9,8/9,5/9]])
        W.append([[-0.8611363116,-0.3399810436,0.3399810436,0.8611363116],
                 [0.3478548451,0.6521451549,0.6521451549,0.3478548451]])
        W.append([[-0.9061798459,-0.5384693101,0,-0.5384693101,-0.9061798459],
                 [0.2369268851,0.4786286705,0.5688888889,0.4786286705,0.2369268851]])
        W.append([[-0.9324695142,-0.6612093865,-0.2386191861,0.2386191861,0.6612093865,0.9324695142],
                 [0.1713244924,0.3607615730,0.4679139346,0.4679139346,0.3607615730,0.1713244924]])
    if option == 'Laguerre':
        W.append([])
        W.append([[0.585786376,3.4142135624],[0.8535533906,0.1464466094]])
        W.append([[0.41577445568,2.2942803603,6.2899450829],[0.7110930099,0.2785177336,0.0103892565]])
        W.append([[0.3225476896,1.7457611012,4.5366202969,9.3950709123],
                 [0.6031541043,0.3574186924,0.0388879085,0.0005392947]])
        W.append([[0.2635603197,1.4134030591,3.5964257710,7.0858100059,12.6408008443],
                 [0.5217556106,0.3986668110,0.0759424497,0.0036117587,0.0000233700]])
    if option == 'Hermite':
        W.append([[0],[1.7724538509]])
        W.append([[0.7071067812,-0.7071067812],[0.8862269255,0.8862269255]])
        W.append([[-1.2247448714,0,1.2247448714],[0.2954089752,1.1816359006,0.2954089752]])
        W.append([[-1.6506801239,-0.5246476233,0.5246476233,1.6506801239],
                  [0.0813128354,0.8049140900,0.8049140900,0.0813128354]])
        W.append([[-2.0201828705,-0.9585724646,0,0.9585724646,2.0201828705],
                  [0.0199532421,0.3936193232,0.9453087205,0.3936193232,0.0199532421]])
        W.append([[-2.3506049737,-1.3358490740,-0.4360774119,0.4360774119,1.3358490740,2.3506049737],
                  [0.0045300100,0.1570673203,0.7246295952,0.7246295952,0.1570673203,0.0045300100]])
        W.append([[-2.6519613568,-1.6735516288,-0.8162878829,0,0.8162878829,1.6735516288,2.6519613568],
                  [0.0009717812,0.0545155828,0.4256072526,0.8102646176,0.4256072526,0.0545155828,0.0009717812]])
        W.append([[-2.9306374203,-1.9816567567,-1.1571937124,-0.3811869902,0.3811869902,1.1571937124,1.9816567567,2.9306374203],
                  [0.0001996041,0.0170779830,0.2078023258,0.6611470126,0.6611470126,0.2078023258,0.0170779830,0.0001996041]])
    return W

def is_symmetric_matrix(A):  #对称矩阵
    m = A.shape[0]
    B = A - A.T
    if (abs(B) > 0).any():
        return False
    return True

def compute_V(f, a):  #计算变号数
    x = symbols('x')
    m = len(f)
    V = 0
    num = f[0].subs({x:a})
    for i in range(1, m):
        temp_num = f[i].subs({x:a})
        if temp_num*num < 0:
            V += 1
            num = temp_num
    return V

def creat_Explicit_Adam_coefficient():
    W = np.array([[1, 1, 0, 0, 0, 0, 0],
                  [2, 3, -1, 0, 0, 0, 0],
                  [12, 23, -16, 5, 0, 0, 0],
                  [24, 55, -59, 37, -9, 0, 0],
                  [720, 1901, -2744, 2616, -1274, 251, 0],
                  [1440, 4277, -7923, 9482, -6798, 2627, -425]])
    return W

def creat_Gear_coefficient():
    alpha = np.array([[1, -1, 0, 0, 0, 0, 0],
                       [1, -3/4, 1/3, 0, 0, 0, 0],
                       [1, -18/11, 9/11, -1/6, 0, 0, 0],
                       [1, -48/25, 36/25, -16/25, 3/25, 0, 0],
                       [1, -300/137, 300/137, -200/137, 75/137, -12/138, 0],
                       [1, -360/147, 450/147, -400/147, 225/147, -72/147, 10/147]])
    beta = np.array([1, 2/3, 6/11, 12/25, 60/137, 60/147])
    return alpha, beta

