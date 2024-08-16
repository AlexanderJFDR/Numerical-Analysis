import matplotlib.pyplot as plt
import numpy as np
from sympy import *

from numerical_analysis.linar_equtions import catch_up, crout_decomposition, guass
from numerical_analysis.supplement import difference_quotient, draw_picture, creat_p, creat_l, draw_piecewised_picture, \
    draw_approximation_curve, creat_Legendre, creat_Chebyshev1, creat_Chebyshev2, creat_Laguerre, creat_Hermite, \
    draw_polynomial_fitting_curve, draw_frequency_domain_diagram


# 1.lagrange插值多项式法
def lagrange_interpolation(a, b):
    m = len(a)
    x = symbols('x')
    y = 0
    for i in range(m):
        L = 1
        for j in range(m):
            if j != i:
                L = L*(x-a[j])/(a[i]-a[j])
        y += b[i]*L
    draw_picture(a, b, y)
    return y

# 2.Newton插值法
def newton_interpolation(a, b):
    m = len(a)
    D = difference_quotient(a, b)
    x = symbols('x')
    y = 0
    for i in range(m):
        L = 1
        for j in range(i):
            L = L*(x-a[j])
        y += D[i, i]*L
    draw_picture(a, b, y)
    return y

# 3.Hermite插值法
def hermite_interpolation(a, b, d):
    n = len(a)
    r = len(d)
    x = symbols('x')
    y = 0
    for j in range(n):
        if j < r:
            l_jn = creat_l(a, j, n)
            l_jr = creat_l(a, j, r)
            h = (1-(x-a[j])*(diff(l_jn,x).subs({x:a[j]}).evalf(6)+diff(l_jr,x).subs({x:a[j]}).evalf(6)))*l_jn*l_jr
            h_ = (x-a[j])*l_jn*l_jr
            y += h*b[j]
            y += h_*d[j]
        else:
            l_jn = creat_l(a, j, n)
            p_r = creat_p(a, r)
            h = l_jn*p_r/p_r.subs({x:a[j]}).evalf(6)
            y += h*b[j]
    draw_picture(a, b, y)
    return y

# 4.分段线性插值法
def piecewise_linear_interpolation(a, b):
    m = len(a)
    x = symbols('x')
    y = []
    for i in range(m-1):
        temp_y = (x-a[i+1])/(a[i]-a[i+1])*b[i]+(x-a[i])/(a[i+1]-a[i])*b[i+1]
        y.append(temp_y)
    draw_piecewised_picture(a, b, y)
    return y

# 5.分段三次Hermite插值法
def piecewise_hermite_interpolation(a, b, d):
    m = len(a)
    x = symbols('x')
    y = []
    for i in range(m-1):
         temp_y = (1+2*(x-a[i])/(a[i+1]-a[i]))*((x-a[i+1])/(a[i]-a[i+1]))**2*b[i]\
                  +(1+2*(x-a[i+1])/(a[i]-a[i+1]))*((x-a[i])/(a[i+1]-a[i]))**2*b[i+1]\
                  +(x-a[i])*((x-a[i+1])/(a[i]-a[i+1]))**2*d[i]\
                  +(x-a[i+1])*((x-a[i])/(a[i+1]-a[i]))**2*d[i+1]
         y.append(temp_y)
    draw_piecewised_picture(a, b, y)
    return y

# 6.三弯矩插值法（条件1）
def cubic_spline_interpolation_1(a, b, df_a=None, df_b=None):
    m = len(a)
    h = np.diff(a)
    if df_a == None and df_b == None:
        df_a = (b[1]-b[0])/h[0]
        df_b = (b[m-1]-b[m-2])/h[m-2]
    A = np.zeros((m, m))
    d = np.zeros((m, 1))
    for i in range(m):
        if i == 0:
            A[i, i] = 2
            A[i, i+1] = 1
            d[i, 0] = 6/h[0]*((b[1]-b[0])/h[0]-df_a)
        elif i == m-1:
            A[i, i-1] = 1
            A[i, i] = 2
            d[i, 0] = 6/h[m-2]*(df_b-(b[m-1]-b[m-2])/h[m-2])
        else:
            A[i, i-1] = h[i-1]/(h[i-1]+h[i])
            A[i, i] = 2
            A[i, i+1] = h[i]/(h[i-1]+h[i])
            d[i, 0] = 6*(((b[i+1]-b[i-1])/(a[i+1]-a[i-1])-(b[i]-b[i-1])/(a[i]-a[i-1]))/(a[i+1]-a[i]))
    M = catch_up(A, d)
    x = symbols('x')
    y = []
    for i in range(m-1):
        temp_y = (a[i+1]-x)**3/6/h[i]*M[i, 0]+(x-a[i])**3/6/h[i]*M[i+1, 0]\
                 +(x-a[i])/h[i]*(b[i+1]-b[i]-h[i]**2/6*(M[i+1, 0]-M[i, 0]))+b[i]-h[i]**2/6*M[i, 0]
        y.append(temp_y)
    draw_piecewised_picture(a, b, y)
    return y

# 7.三弯矩插值法（条件二）
def cubic_spline_interpolation_2(a, b, ddf_a=None, ddf_b=None):
    m = len(a)
    h = np.diff(a)
    if ddf_a == None and ddf_b == None:
        ddf_a = 0
        ddf_b = 0
    A = np.zeros((m-2, m-2))
    d = np.zeros((m-2, 1))
    for i in range(m-2):
        if i == 0:
            A[i, i] = 2
            A[i, i+1] = h[1]/(h[0]+h[1])
            d[i, 0] = 6*(((b[2]-b[0])/(a[2]-a[0])-(b[1]-b[0])/(a[1]-a[0]))/(a[2]-a[1]))-(h[0]/(h[0]+h[1]))*ddf_a
        elif i == m-3:
            A[i, i-1] = h[m-4]/(h[m-4]+h[m-3])
            A[i, i] = 2
            d[i, 0] = 6*(((b[m-1]-b[m-3])/(a[m-1]-a[m-3])-(b[m-2]-b[m-3])/(a[m-2]-a[m-3]))/(a[m-1]-a[m-2]))\
                      -(h[m-3]/(h[m-4]+h[m-3]))*ddf_b
        else:
            A[i, i-1] = h[i]/(h[i]+h[i+1])
            A[i, i] = 2
            A[i, i+1] = h[i+1]/(h[i]+h[i+1])
            d[i, 0] = 6*(((b[i+2]-b[i])/(a[i+2]-a[i])-(b[i+1]-b[i])/(a[i+1]-a[i]))/(a[i+2]-a[i+1]))
    temp_M = catch_up(A, d)
    M = np.zeros((m, 1))
    M[0, 0] = ddf_a
    M[m-1, 0] = ddf_b
    for i in range(m-2):
        M[i+1, 0] = temp_M[i, 0]
    x = symbols('x')
    y = []
    for i in range(m - 1):
        temp_y = (a[i + 1] - x) ** 3 / 6 / h[i] * M[i, 0] + (x - a[i]) ** 3 / 6 / h[i] * M[i + 1, 0] \
                 + (x - a[i]) / h[i] * (b[i + 1] - b[i] - h[i] ** 2 / 6 * (M[i + 1, 0] - M[i, 0])) + b[i] - h[
                     i] ** 2 / 6 * M[i, 0]
        y.append(temp_y)
    draw_piecewised_picture(a, b, y)
    return y

# 8.三弯矩插值法（条件三）
def cubic_spline_interpolation_3(a, b):
    m = len(a)
    if b[0] != b[m-1]:
        print('This method is not applicable.')
        print('The header and tail elements must be the same!')
        return
    h = np.diff(a)
    A = np.zeros((m - 1, m - 1))
    d = np.zeros((m - 1, 1))
    for i in range(m-1):
        if i == 0:
            A[i, i] = 2
            A[i, i+1] = 0.5
            A[i, m-2] = 0.5
            d[i, 0] = 6*(((b[i+2]-b[i])/(a[i+2]-a[i])-(b[i+1]-b[i])/(a[i+1]-a[i]))/(a[i+2]-a[i+1]))
        elif i == m-2:
            A[i, 0] = h[0]/(h[m-2]+h[0])
            A[i, i-1] = h[m-2]/(h[m-2]+h[0])
            A[i, i] = 2
            d[i, 0] = 6*(((b[1]-b[m-2])/(a[1]-a[m-2])-(b[m-1]-b[m-2])/(a[m-1]-a[m-2]))/(a[1]-a[m-1]))
        else:
            A[i, i - 1] = h[i] / (h[i] + h[i + 1])
            A[i, i] = 2
            A[i, i + 1] = h[i + 1] / (h[i] + h[i + 1])
            d[i, 0] = 6*(((b[i+2]-b[i])/(a[i+2]-a[i])-(b[i+1]-b[i])/(a[i+1]-a[i]))/(a[i+2]-a[i+1]))
    temp_M = crout_decomposition(A, d)
    M = np.zeros((m, 1))
    M[0, 0] = temp_M[m-2, 0]
    for i in range(m-1):
        M[i+1, 0] = temp_M[i, 0]
    x = symbols('x')
    y = []
    for i in range(m - 1):
        temp_y = (a[i + 1] - x) ** 3 / 6 / h[i] * M[i, 0] + (x - a[i]) ** 3 / 6 / h[i] * M[i + 1, 0] \
                 + (x - a[i]) / h[i] * (b[i + 1] - b[i] - h[i] ** 2 / 6 * (M[i + 1, 0] - M[i, 0])) + b[i] - h[
                     i] ** 2 / 6 * M[i, 0]
        y.append(temp_y)
    draw_piecewised_picture(a, b, y)
    return y

# 9.三转角插值法（条件一）
def three_angle_interpolation1(a, b, df_a=None, df_b=None):
    m = len(a)
    h = np.diff(a)
    if df_a == None and df_b == None:
        df_a = (b[1]-b[0])/h[0]
        df_b = (b[m-1]-b[m-2])/h[m-2]
    A = np.zeros((m - 2, m - 2))
    d = np.zeros((m - 2, 1))
    for i in range(m - 2):
        if i == 0:
            A[i, i] = 2
            A[i, i + 1] = h[0] / (h[0] + h[1])
            d[i, 0] = 3*((h[1]/(h[0]+h[1]))*((b[1]-b[0])/h[0])+(h[0]/(h[0]+h[1]))*((b[2]-b[1])/h[1]))\
                      -(h[1]/(h[0]+h[1]))*df_a
        elif i == m - 3:
            A[i, i - 1] = h[m - 3] / (h[m - 4] + h[m - 3])
            A[i, i] = 2
            d[i, 0] = 3*((h[m-3]/(h[0]+h[m-3]))*((b[m-2]-b[m-3])/h[m-3])+(h[0]/(h[0]+h[m-3]))*((b[m-1]-b[m-2])/b[m-2]))\
                      -(h[m-3]/(h[m-3]+h[m-2]))*df_b
        else:
            A[i, i - 1] = h[i+1] / (h[i] + h[i + 1])
            A[i, i] = 2
            A[i, i + 1] = h[i] / (h[i] + h[i + 1])
            d[i, 0] = 3*((h[i+1]/(h[i]+h[i+1]))*((b[i+1]-b[i])/h[i])+(h[i]/(h[i]+h[i+1]))*((b[i+2]-b[i+1])/b[i+1]))
    temp_M = catch_up(A, d)
    M = np.zeros((m, 1))
    M[0, 0] = df_a
    M[m-1, 0] = df_b
    for i in range(m - 2):
        M[i + 1, 0] = temp_M[i, 0]
    x = symbols('x')
    y = []
    for i in range(m - 1):
        temp_y = ((x-a[i+1])**2*(h[i]+2*(x-a[i]))/h[i]**3)*b[i]+((x-a[i])**2*(h[i]+2*(a[i+1]-x))/h[i]**3)*b[i+1]\
                 +((x-a[i+1])**2*(x-a[i])/h[i]**2)*M[i, 0]+((x-a[i])**2*(x-a[i+1])/h[i]**2)*M[i+1, 0]
        y.append(temp_y)
    draw_piecewised_picture(a, b, y)
    return y

# 10.三转角插值法（条件二）
def three_angle_interpolation2(a, b, ddf_a=None, ddf_b=None):
    m = len(a)
    h = np.diff(a)
    if ddf_a == None and ddf_b == None:
        ddf_a = 0
        ddf_b = 0
    A = np.zeros((m, m))
    d = np.zeros((m, 1))
    for i in range(m):
        if i == 0:
            A[i, i] = 2
            A[i, i + 1] = 1
            d[i, 0] = 3*(a[1]-a[0])/h[0]-h[0]/2*ddf_a
        elif i == m - 1:
            A[i, i - 1] = 1
            A[i, i] = 2
            d[i, 0] = 3*(a[m-1]-a[m-2])/h[m-2]-h[m-2]/2*ddf_b
        else:
            A[i, i - 1] = h[i] / (h[i-1] + h[i])
            A[i, i] = 2
            A[i, i + 1] = h[i-1] / (h[i-1] + h[i])
            d[i, 0] = 3*((h[i]/(h[i-1]+h[i]))*((b[i]-b[i-1])/h[i-1])+(h[i-1]/(h[i-1]+h[i]))*(b[i+1]-b[i])/h[i])
    M = catch_up(A, d)
    x = symbols('x')
    y = []
    for i in range(m - 1):
        temp_y = ((x-a[i+1])**2*(h[i]+2*(x-a[i]))/h[i]**3)*b[i]+((x-a[i])**2*(h[i]+2*(a[i+1]-x))/h[i]**3)*b[i+1]\
                 +((x-a[i+1])**2*(x-a[i])/h[i]**2)*M[i, 0]+((x-a[i])**2*(x-a[i+1])/h[i]**2)*M[i+1, 0]
        y.append(temp_y)
    draw_piecewised_picture(a, b, y)
    return y

# 11.三转角插值法（条件三）
def three_angle_interpolation3(a, b):
    m = len(a)
    if b[0] != b[m - 1]:
        print('This method is not applicable.')
        print('The header and tail elements must be the same!')
        return
    h = np.diff(a)
    A = np.zeros((m - 1, m - 1))
    d = np.zeros((m - 1, 1))
    for i in range(m - 1):
        if i == 0:
            A[i, i] = 2
            A[i, i + 1] = 0.5
            A[i, m - 2] = 0.5
            d[i, 0] = 3*((h[1]/(h[0]+h[1]))*((b[1]-b[0])/h[0])+(h[0]/(h[0]+h[1]))*((b[2]-b[1])/h[1]))
        elif i == m - 2:
            A[i, 0] = h[m-2]/(h[m-2]+h[0])
            A[i, i - 1] = h[0]/(h[m-2]+h[0])
            A[i, i] = 2
            d[i, 0] = 3*((h[m-3]/(h[m-3]+h[m-2]))*((b[1]-b[0])/h[0])+(h[m-2]/(h[m-3]+h[m-2]))*(b[m-1]-b[m-2])/h[m-2])
        else:
            A[i, i - 1] = h[i+1] / (h[i] + h[i + 1])
            A[i, i] = 2
            A[i, i + 1] = h[i] / (h[i] + h[i + 1])
            d[i, 0] = 3*((h[i+1]/(h[i]+h[i+1]))*((b[i+1]-b[i])/h[i])+(h[i]/(h[i]+h[i+1]))*((b[i+2]-b[i+1])/b[i+1]))
    temp_M = crout_decomposition(A, d)
    M = np.zeros((m, 1))
    M[0, 0] = temp_M[m - 2, 0]
    for i in range(m - 1):
        M[i + 1, 0] = temp_M[i, 0]
    x = symbols('x')
    y = []
    for i in range(m - 1):
        temp_y = ((x - a[i + 1]) ** 2 * (h[i] + 2 * (x - a[i])) / h[i] ** 3) * b[i] + (
                    (x - a[i]) ** 2 * (h[i] + 2 * (a[i + 1] - x)) / h[i] ** 3) * b[i + 1] \
                 + ((x - a[i + 1]) ** 2 * (x - a[i]) / h[i] ** 2) * M[i, 0] + (
                             (x - a[i]) ** 2 * (x - a[i + 1]) / h[i] ** 2) * M[i + 1, 0]
        y.append(temp_y)
    draw_piecewised_picture(a, b, y)
    return y

# 12.有理逼近法
def rational_approximation(a, b):
    m = len(a)
    v = np.zeros((m, m))
    for j in range(m):
        for i in range(j, m):
            if j == 0:
                v[i, j] = b[i]
            else:
                v[i, j] = (a[i]-a[j-1])/(v[i, j-1]-v[j-1, j-1])
    x = symbols('x')
    y = v[m-1, m-1]
    for i in reversed(range(m-1)):
        y = v[i, i]+(x-a[i])/y
    draw_picture(a, b, y)
    return y

# 13.线性无关函数族的最佳函数逼近
def best_square_approximation(g, f, C, w=1, n=6):
    m = len(f)
    (a, b) = C
    x = symbols('x')
    G = np.zeros((m, m))
    d = np.zeros((m, 1))
    for i in range(m):
        for j in range(m):
            G[i, j] = integrate(w*f[i]*f[j], (x, a, b)).evalf(n)
        d[i, 0] = integrate(w*f[i]*g, (x, a, b)).evalf(n)
    x = guass(G, d)
    y = 0
    for i in range(m):
        y += x[i, 0]*f[i]
    draw_approximation_curve(g, y, a, b)
    return y

# 14.正交函数族的最佳函数逼近
def best_square_approximation_optimal(g, m, option=None, n=6):
    x = symbols('x')
    if option == 'Legendre' or option == None:
        f = creat_Legendre(m)
        w = 1
        (a, b) = (-1, 1)
    elif option == 'Chebyshev1':
        f = creat_Chebyshev1(m)
        w = 1/sqrt(1-x**2)
        (a, b) = (-1, 1)
    elif option == 'Chebyshev2':
        f = creat_Chebyshev2(m)
        w = sqrt(1 - x ** 2)
        (a, b) = (-1, 1)
    elif option == 'Laguerre':
        f = creat_Laguerre(m)
        w = exp(-x)
        (a, b) = (0, float('inf'))
    elif option == 'Hermite':
        f = creat_Hermite(m)
        w = exp(-x**2)
        (a, b) = (-float('inf'), float('inf'))
    else:
        print('Please enter the correct option')
        print('Optional options: Legendre, Chebyshev1, Chebyshev2, Laguerre, Hermite')
        return
    y = 0
    for i in range(m):
        t1 = integrate(w*f[i]*g, (x, a, b)).evalf(n)
        t2 = integrate(w*f[i]*f[i], (x, a, b)).evalf(n)
        y += (t1/t2)*f[i]
    if option == 'Laguerre':
        draw_approximation_curve(g, y, 0, 1)
    else:
        draw_approximation_curve(g, y, -1, 1)
    return y

# 15.多项式曲线拟合
# w 权重，degree 多项式次数，alpha 平滑因子
def polynomial_curve_fitting(a, b, degree, w=None, alpha=0, n=6):
    x = symbols('x')
    m = len(a)
    if w == None:
        w = np.linspace(1, 1, m)
    f = []
    ddf = []
    for i in range(degree):
        f.append(x**i)
        ddf.append(diff(x**i, x, 2))
    f_a = np.zeros((degree, m))
    for i in range(degree):
        for j in range(m):
            f_a[i, j] = f[i].subs({x: a[j]}).evalf(n)
    G = np.zeros((degree, degree))
    d = np.zeros((degree, 1))
    for i in range(degree):
        for j in range(degree):
            G[i, j] = np.sum(w*f_a[i, :]*f_a[j, :])+alpha*integrate(ddf[j]*ddf[i], (x, min(a), max(a))).evalf(n)
        d[i, 0] = np.sum(w*b*f_a[i, :])
    k = guass(G, d)
    y = 0
    for i in range(degree):
        y += k[i, 0]*f[i]
    draw_polynomial_fitting_curve(a, b, y)
    delta2 = np.sum(w*b*b)-np.dot(k.T, np.dot(f_a, np.array(b).reshape(m, 1)))
    sigma2 = float(delta2/(m-degree-1))
    return y, sigma2

# 16.选择最佳拟合次数
def select_fit_times(a, b, top_times, w=None, alpha=0, n=6):
    sigma2 = np.linspace(0, 0, top_times)
    for i in range(1, top_times+1):
        y, sigma2[i-1] = polynomial_curve_fitting(a, b, i, w, alpha, n)
    t = np.linspace(1, top_times, top_times)
    print(sigma2)
    plt.plot(t, sigma2)
    plt.show()

# 17.周期函数逼近
def periodic_function_approximation(f, degree, n=6):
    x = symbols('x')
    a = np.linspace(0, 0, degree)
    b = np.linspace(0, 0, degree)
    a_0 = 1/pi*integrate(f, (x, 0, 2*pi))
    for i in range(degree):
        a[i] = 1/pi*integrate(f*cos((i+1)*x), (x, 0, 2*pi))
        b[i] = 1/pi*integrate(f*sin((i+1)*x), (x, 0, 2*pi))
    y = a_0/2
    for i in range(degree):
        y += a[i]*cos((i+1)*x)+b[i]*sin((i+1)*x)
    draw_approximation_curve(f, y, 0, float(2*pi))
    draw_frequency_domain_diagram(a_0, a, b)
    return y

# 18.DFT算法
def discrete_Fourier_transformation(data, n=6):
    m = len(data)
    x = symbols('x')
    a = np.linspace(0, 0, m)
    b = np.linspace(0, 0, m)
    a_0 = 2/m*np.sum(data)
    for i in range(m):
        for j in range(m):
            a[i] += 2/m*data[j]*cos(2*pi*(i+1)*j/m)
            b[i] += 2/m*data[j]*sin(2*pi*(i+1)*j/m)
    y = a_0/2
    for i in range(m):
        y += a[i]*cos((i+1)*x)+b[i]*sin((i+1)*x)
    t = np.linspace(0, float(2*pi), m)
    draw_polynomial_fitting_curve(t, data, y)
    draw_frequency_domain_diagram(a_0, a, b)
    return y