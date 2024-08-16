import numpy as np
from sympy import *
from numerical_analysis.supplement import creat_Newton_Cotes_coefficient, creat_Gauss_coefficient


# 1.Newton-Cotes公式
def newton_cotes_integration(f, C, degree, subinterval_num=100):
    x = symbols('x')
    (a, b) = C
    W = creat_Newton_Cotes_coefficient()
    y = 0
    for i in range(subinterval_num):
        for j in range(degree+1):
            y += (b-a)/subinterval_num*W[degree-1, 0]*W[degree-1, j+1]*f.subs({x: a+(b-a)/subinterval_num*(i+j/degree)})
    return y

# 2.Romberg积分法
def romberg_integration(f, C, max_degree=10):
    x = symbols('x')
    (a, b) = C
    T = np.zeros((max_degree+1, max_degree+1))
    i = 0
    T[0, 0] = (b-a)/2*(f.subs({x:a})+f.subs({x:b}))
    while i<max_degree:
        i += 1
        sum = 0
        for j in range(1, 2**(i-1)+1):
            sum += f.subs({x:a+(2*j-1)*(b-a)/2**i})
        T[0, i] = 0.5*(T[0, i-1]+(b-a)/2**(i-1)*sum)
        for j in range(1, i+1):
            T[j, i-j] = (4**j*T[j-1, i-j+1]-T[j-1, i-j])/(4**j-1)
    print(T)
    return T[i, 0]

# 3.Gauss求积公式
def gauss_integration(f, C, degree, option='Legendre'):
    x = symbols('x')
    (a, b) = C
    y = 0
    if option == 'Legendre':
        W = creat_Gauss_coefficient(option)
        for i in range(degree):
            y += (b-a)/2*np.array(W[degree-1])[1,i]*f.subs({x:(a+b)/2+(b-a)/2*np.array(W[degree-1])[0,i]})
    elif option == 'Laguerre':
        W = creat_Gauss_coefficient(option)
        g = f*exp(x)
        if a=='-inf':
            for i in range(degree):
                y += np.array(W[degree-1])[1,i]*g.subs({x:b-np.array(W[degree-1])[0,i]})
        elif b=='inf':
            for i in range(degree):
                y += np.array(W[degree-1])[1,i]*g.subs({x:a+np.array(W[degree-1])[0,i]})
    elif option == 'Hermite':
        W = creat_Gauss_coefficient(option)
        g = exp(x**2)*f
        for i in range(degree):
            y += np.array(W[degree-1])[1,i]*g.subs({x:np.array(W[degree-1])[0,i]})
    else:
        print('Please enter the correct option')
        print('Optional options: Legendre, Laguerre, Hermite')
    return y

# 4.复化的高斯型求积公式
def complex_gauss_integration(f, C, degree, subinterval_num=100):
    x = symbols('x')
    (a, b) = C
    W = creat_Gauss_coefficient('Legendre')
    y = 0
    for j in range(degree):
        for i in range(subinterval_num):
            y += 0.5*np.array(W[degree-1])[1,j]*(b-a)/subinterval_num*\
                 f.subs({x:(a+(b-a)*(2*i+1)/subinterval_num/2+(b-a)/subinterval_num/2*np.array(W[degree-1])[0,j])})
    return y

# 5.自适应积分方法
def adaptive_integration(f, C, epsilon=1e-6):
    x = symbols('x')
    (a, b) = C
    H = b-a
    point = a
    r=0
    y = 0
    while point!=b:
        if abs(newton_cotes_integration(f, (point,point+H/2**r), 3, subinterval_num=2)-
               newton_cotes_integration(f, (point,point+H/2**r), 3, subinterval_num=1)) <= 15/2**r*epsilon:
            y += newton_cotes_integration(f, (point,point+H/2**r), 3, subinterval_num=2)
            point = point+H/2**r
        else:
            r += 1
    return y

