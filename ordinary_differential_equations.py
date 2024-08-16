import numpy as np
from matplotlib import pyplot as plt
from sympy import symbols

from numerical_analysis.linar_equtions import catch_up, crout_decomposition
from numerical_analysis.supplement import creat_Explicit_Adam_coefficient, creat_Gear_coefficient


# Euler方法
def euler_method(f, C, initial_point, step=0.01, is_draw=True):
    x = symbols('x')
    y = symbols('y')
    (a, b) = C
    (x0, y0) = initial_point
    num_point = int(np.floor((b-a)/step+1e-9))
    X = np.linspace(0, 0, num_point)
    Y = np.linspace(0, 0, num_point)
    if a >= x0:
        h = step
        temp_x = x0
        temp_y = y0
        i = 0
        while temp_x<b:
            if temp_x>=a and i<num_point:
                X[i] = temp_x
                Y[i] = temp_y
                i += 1
            temp_y = temp_y+h*f.subs({x:temp_x, y:temp_y})
            temp_x += h
    elif b <= x0:
        h = -1*step
        temp_x = x0
        temp_y = y0
        i = num_point-1
        while temp_x >= a:
            if temp_x < b and i>=0:
                X[i] = temp_x
                Y[i] = temp_y
                i -= 1
            temp_y = temp_y + h * f.subs({x: temp_x, y: temp_y})
            temp_x += h
    else:
        h = -1 * step
        temp_x = x0 + h
        temp_y = y0 + h * f.subs({x: x0, y: y0})
        i = int(np.floor((x0-a)/step-1))
        while temp_x >= a and i>=0:
            X[i] = temp_x
            Y[i] = temp_y
            i -= 1
            temp_y = temp_y + h * f.subs({x: temp_x, y: temp_y})
            temp_x += h
        X[int(np.floor((x0-a)/step))] = x0
        Y[int(np.floor((x0-a)/step))] = y0
        h = step
        temp_x = x0 + h
        temp_y = y0 + h * f.subs({x: x0, y: y0})
        i = int(np.floor((x0-a)/step+1))
        while temp_x < b and i<num_point:
            X[i] = temp_x
            Y[i] = temp_y
            i += 1
            temp_y = temp_y + h * f.subs({x: temp_x, y: temp_y})
            temp_x += h
    if is_draw==True:
        plt.plot(X, Y, 'r', linewidth=2)
        plt.show()
    return X, Y

# 2.Milne方法
def milne_method(f, C, initial_point, step=0.01, is_draw=True):
    x = symbols('x')
    y = symbols('y')
    (a, b) = C
    h = step
    num_point = int(np.floor((b - a) / step+1e-9))
    temp_X, temp_Y = runge_kutte_method(f, (a, a+step*4), initial_point, degree=4, step=step, is_draw=False)
    temp_f2 = f.subs({x:temp_X[1],y:temp_Y[1]})
    temp_f1 = f.subs({x: temp_X[2], y: temp_Y[2]})
    temp_f0 = f.subs({x: temp_X[3], y: temp_Y[3]})
    temp_y3 = b
    X = np.linspace(0, 0, num_point)
    Y = np.linspace(0, 0, num_point)
    X[0:4] = temp_X
    Y[0:4] = temp_Y
    for i in range(4, num_point):
        X[i] = X[i-1]+h
        Y[i] = temp_y3+4*h/3*(2*temp_f0-temp_f1+2*temp_f2)
        temp_y3 = Y[i-3]
        temp_f2 = temp_f1
        temp_f1 = temp_f0
        temp_f0 = f.subs({x:X[i], y:Y[i]})
    if is_draw == True:
        plt.plot(X, Y, 'r', linewidth=2)
        plt.show()
    return X, Y

# 3.显式Adam方法
def explicit_adam_method(f, C, initial_point, degree, step=0.01, is_draw=True):
    x = symbols('x')
    y = symbols('y')
    (a, b) = C
    h = step
    num_point = int(np.floor((b - a) / step+1e-9))
    temp_f = np.linspace(0, 0, degree)
    temp_X, temp_Y = runge_kutte_method(f, (a, a+step*degree), initial_point, degree=degree, step=step, is_draw=False)
    for i in range(degree):
        temp_f[degree-i-1] = f.subs({x:temp_X[i], y:temp_Y[i]})
    W = creat_Explicit_Adam_coefficient()
    X = np.linspace(0, 0, num_point)
    Y = np.linspace(0, 0, num_point)
    X[0:degree] = temp_X
    Y[0:degree] = temp_Y
    for i in range(degree, num_point):
        X[i] = X[i-1]+h
        temp_num = 0
        for j in range(degree):
            temp_num += W[degree-1, j+1]*temp_f[j]
        Y[i] = Y[i-1]+h/W[degree-1, 0]*temp_num
        for k in range(degree-1):
            temp_f[degree - k - 1] = temp_f[degree-k-2]
        temp_f[0] = f.subs({x:X[i], y:Y[i]})
    if is_draw == True:
        plt.plot(X, Y, 'r', linewidth=2)
        plt.show()
    return X, Y

# 4.Runge-Kutta方法
def runge_kutte_method(f, C, initial_point, degree=4, option=None, step=0.01, is_draw=True):
    x = symbols('x')
    y = symbols('y')
    (a, b) = C
    h = step
    num_point = int(np.floor((b - a) / step+1e-9))
    X = np.linspace(0, 0, num_point)
    Y = np.linspace(0, 0, num_point)
    (X[0], Y[0]) = initial_point
    if degree == 2:
        if option == 'midpoint' or option == None:
            for i in range(1, num_point):
                X[i] = X[i-1]+h
                Y[i] = Y[i - 1] + h * f.subs({x: X[i-1]+h/2, y: Y[i-1]+h/2*f.subs({x:X[i-1], y:Y[i-1]})})
        elif option == 'Heun':
            for i in range(1, num_point):
                X[i] = X[i-1]+h
                Y[i] = Y[i-1]+h/4*(f.subs({x:X[i-1], y:Y[i-1]})+3*f.subs({x:X[i-1]+2/3*h,y:Y[i-1]+2/3*h*f.subs({x:X[i-1], y:Y[i-1]})}))
        elif option == 'updated_Euler':
            for i in range(1, num_point):
                X[i] = X[i-1]+h
                Y[i] = Y[i-1]+h/2*(f.subs({x:X[i-1], y:Y[i-1]})+f.subs({x:X[i-1]+h, y:Y[i-1]+h*f.subs({x:X[i-1], y:Y[i-1]})}))
        else:
            print('option only accept midpoint, Heun, updated_Euler')
            return
    elif degree == 3:
        if option == 'best' or option == None:
            for i in range(1, num_point):
                X[i] = X[i-1]+h
                K1 = f.subs({x:X[i-1], y:Y[i-1]})
                K2 = f.subs({x:X[i-1]+h/2, y:Y[i-1]+h/2*K1})
                K3 = f.subs({x:X[i-1]+3/4*h, y:Y[i-1]+3/4*h*K2})
                Y[i] = Y[i-1]+h/9*(2*K1+3*K2+4*K3)
        else:
            print('option only accept best')
            return
    elif degree == 4:
        if option == 'normal' or option == None:
            for i in range(1, num_point):
                X[i] = X[i-1]+h
                K1 = f.subs({x:X[i-1], y:Y[i-1]})
                K2 = f.subs({x:X[i-1]+h/2, y:Y[i-1]+h/2*K1})
                K3 = f.subs({x:X[i-1]+h/2, y:Y[i-1]+h/2*K2})
                K4 = f.subs({x:X[i-1]+h, y:Y[i-1]+h*K3})
                Y[i] = Y[i-1]+h/6*(K1+2*K2+2*K3+K4)
        elif option == 'best':
            for i in range(1, num_point):
                X[i] = X[i-1]+h
                K1 = f.subs({x:X[i-1], y:Y[i-1]})
                K2 = f.subs({x:X[i-1]+0.4*h, y:Y[i-1]+0.4*h*K1})
                K3 = f.subs({x:X[i-1]+0.45573725*h, y:Y[i-1]+0.29697761*h*K1+0.15875964*h*K2})
                K4 = f.subs({x:X[i-1]+h, y:Y[i-1]+0.21810040*h*K1-3.05096516*h*K2+3.83286476*h*K3})
                Y[i] = Y[i-1]+h*(0.17476028*K1-0.55148066*K2+1.20553560*K3+0.17118478*K4)
        else:
            print('option only accept normal, best')
            return
    else:
        print('degree only accept 2, 3, 4')
        return
    if is_draw == True:
        plt.plot(X, Y, 'r', linewidth=2)
        plt.show()
    return X, Y

# 5.预测-校正方法
def prediction_correction_method(f, C, initial_point, option='Adams', mode='PMECME', step=0.01, is_draw=True):
    x = symbols('x')
    y = symbols('y')
    (a, b) = C
    h = step
    num_point = int(np.floor((b - a) / step+1e-9))
    X = np.linspace(0, 0, num_point)
    Y = np.linspace(0, 0, num_point)
    if option == 'Euler':
        (X[0], Y[0]) = initial_point
        for i in range(1, num_point):
            X[i] = X[i-1]+h
            temp_y = Y[i-1]+h*f.subs({x:X[i-1], y:Y[i-1]})
            Y[i] = Y[i-1]+h/2*(f.subs({x:X[i-1], y:Y[i-1]})+f.subs({x:X[i], y:temp_y}))
    elif option == 'Adams':
        temp_f = np.linspace(0, 0, 4)
        temp_X, temp_Y = runge_kutte_method(f, (a, a + step * 4), initial_point, degree=4, step=step, is_draw=False)
        X[:4] = temp_X
        Y[:4] = temp_Y
        for i in range(4):
            temp_f[3-i] = f.subs({x: temp_X[i], y: temp_Y[i]})
        if mode =='PECE':
            for i in range(4, num_point):
                X[i] = X[i-1]+h
                pre_y0 = Y[i-1]+h/24*(55*temp_f[0]-59*temp_f[1]+37*temp_f[2]-9*temp_f[3])
                pre_f0 = f.subs({x:X[i], y:pre_y0})
                Y[i] = Y[i-1]+h/24*(9*pre_f0+19*temp_f[0]-5*temp_f[1]+temp_f[2])
                for k in range(3):
                    temp_f[3-k] = temp_f[2-k]
                temp_f[0] = f.subs({x: X[i], y: Y[i]})
        elif mode == 'PMECME':
            temp_sub = 0
            for i in range(4, num_point):
                X[i] = X[i-1]+h
                pre_y0 = Y[i-1]+h/24*(55*temp_f[0]-59*temp_f[1]+37*temp_f[2]-9*temp_f[3])
                exp_y0 = pre_y0+251/270*temp_sub
                pre_f0 = f.subs({x:X[i], y:exp_y0})
                pre_y1 = Y[i-1]+h/24*(9*pre_f0+19*temp_f[0]-5*temp_f[1]+temp_f[2])
                Y[i] = pre_y1-19/270*(pre_y1-pre_y0)
                temp_sub = Y[i]-pre_y0
                for k in range(3):
                    temp_f[3-k] = temp_f[2-k]
                temp_f[0] = f.subs({x: X[i], y: Y[i]})
        else:
            print('mode only accept PECE, PMECME')
            return
    elif option == 'Miline_Hamming':
        temp_f = np.linspace(0, 0, 3)
        temp_X, temp_Y = runge_kutte_method(f, (a, a + step * 4), initial_point, degree=4, step=step, is_draw=False)
        X[:4] = temp_X
        Y[:4] = temp_Y
        for i in range(3):
            temp_f[i] = f.subs({x: temp_X[3-i], y:temp_Y[3-i]})
        if mode == 'PECE':
            for i in range(4, num_point):
                X[i] = X[i-1]+h
                pre_y0 = Y[i-4]+4/3*h*(2*temp_f[0]-temp_f[1]+2*temp_f[2])
                pre_f0 = f.subs({x:X[i], y:pre_y0})
                Y[i] = (9*Y[i-1]-Y[i-3])/8+3/8*h*(pre_f0+2*temp_f[0]-temp_f[1])
                for k in range(2):
                    temp_f[2-k] = temp_f[1-k]
                temp_f[0] = f.subs({x: X[i], y: Y[i]})
        elif mode == 'PMECME':
            temp_sub = 0
            for i in range(4, num_point):
                X[i] = X[i - 1] + h
                pre_y0 = Y[i-4]+4/3*h*(2*temp_f[0]-temp_f[1]+2*temp_f[2])
                exp_y0 = pre_y0+112/121*temp_sub
                pre_f0 = f.subs({x: X[i], y: exp_y0})
                pre_y1 = (9*Y[i-1]-Y[i-3])/8+3/8*h*(pre_f0+2*temp_f[0]-temp_f[1])
                Y[i] = pre_y1-9/121*(pre_y1-pre_y0)
                temp_sub = Y[i]-pre_y0
                for k in range(2):
                    temp_f[2-k] = temp_f[1-k]
                temp_f[0] = f.subs({x: X[i], y: Y[i]})
        else:
            print('mode only accept PECE, PMECME')
            return
    else:
        print('option only accept Euler, Adams, Miline_Hamming')
    if is_draw == True:
        plt.plot(X, Y, 'r', linewidth=2)
        plt.show()
    return X, Y

# 6.方程组求解 runge_kutte4
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
        for i in range(m):
            plt.plot(X[i, :], Y[i, :], 'r', linewidth=2)
        plt.show()
    return X, Y

# 7.方程组求解 Milne_Hamming
def differential_equations_Milne_Hamming(f, C, initial_point, step=0.01, is_draw=True):
    m = len(f)
    x = symbols('x1:%d' % (m + 1))
    y = symbols('y1:%d' % (m + 1))
    [a, b] = C
    num_point = int(np.floor((b - a) / step+1e-9))
    h = (b - a) / num_point
    X = np.zeros((m, num_point))
    Y = np.zeros((m, num_point))
    temp_f = np.zeros((m, 3))
    temp_X, temp_Y = differential_equations_runge_kutte4(f, (a, a + step * 4), initial_point, step=step, is_draw=False)
    X[:, 0:4] = temp_X
    Y[:, 0:4] = temp_Y
    for i in range(m):
        for j in range(3):
            temp_f[i, j] = f[i].subs(dict(zip([*x, *y], [*temp_X[:, j+1], *temp_Y[:, j+1]])))
    for i in range(4, num_point):
        X[:, i] = X[:, i-1]+h
        pre_y0 = Y[:, i-4] + 4 / 3 * h * (2 * temp_f[:, 0] - temp_f[:, 1] + 2 * temp_f[:, 2])
        pre_f0 = np.linspace(0,0,m)
        for j in range(m):
            pre_f0[j] = f[j].subs(dict(zip([*x, *y],[*X[:, i], *pre_y0])))
        Y[:, i] = (9 * Y[:, i - 1] - Y[:, i - 3]) / 8 + 3 / 8 * h * (pre_f0 + 2 * temp_f[:, 0] - temp_f[:, 1])
        for k in range(2):
            temp_f[:, 2 - k] = temp_f[:, 1 - k]
        for k in range(m):
            temp_f[k, 0] = f[k].subs(dict(zip([*x, *y],[*X[:, i], *Y[:, i]])))
    if is_draw == True:
        for i in range(m):
            plt.plot(X[i, :], Y[i, :], 'r', linewidth=2)
        plt.show()
    return X, Y

# 8.Gear方法
def gear_method(f, C, initial_point, degree, step=0.01, is_draw=True):
    m = len(f)
    x = symbols('x1:%d' % (m + 1))
    y = symbols('y1:%d' % (m + 1))
    [a, b] = C
    num_point = int(np.floor((b - a) / step+1e-9))
    h = step
    X = np.zeros((m, num_point))
    Y = np.zeros((m, num_point))
    temp_x1, temp_y1 = differential_equations_runge_kutte4(f, (a, a+(degree+1)*h), initial_point, step=h, is_draw=False)
    temp_x = np.zeros((m, degree+1))
    temp_y = np.zeros((m, degree+1))
    for i in range(degree+1):
        temp_x[:, i] = temp_x1[:, degree-i]
        temp_y[:, i] = temp_y1[:, degree-i]
    X[:, 0:degree+1] = temp_x1
    Y[:, 0:degree+1] = temp_y1
    alpha, beta = creat_Gear_coefficient()
    for i in range(degree, num_point):
        X[:, i] = X[:, i-1]+h
        temp_f = np.linspace(0,0,m)
        for j in range(m):
            temp_f[j] = f[j].subs(dict(zip([*x, *y], [*X[:, i], *Y[:, i]])))
        temp_num = 0
        for k in range(1, degree+1):
            temp_num += alpha[degree-1, k]*temp_y[:, k]
        Y[:, i] = (h*beta[degree-1]*temp_f-temp_num)/alpha[degree-1, 0]
        for k in range(degree):
            temp_y[:, degree-k] = temp_y[:, degree-1-k]
        temp_initial_point = np.zeros((m, 2))
        for k in range(m):
            temp_initial_point[k, 0] = X[k, i]
            temp_initial_point[k, 1] = Y[k, i]
        _, temp_Y = differential_equations_runge_kutte4(f, (X[0, i], X[0, i]+2*h), temp_initial_point, step=h, is_draw=False)
        for k in range(m):
            temp_y[k, 0] = temp_Y[k, 1]
        if i < num_point-1:
            Y[:, i+1] = temp_y[:, 0]
    if is_draw == True:
        for i in range(m):
            plt.plot(X[i, :], Y[i, :], 'r', linewidth=2)
        plt.show()
    return X, Y

# 9.打靶法
def shooting_method(f, C, initial_point, initial_z, type=1, iteration=10, epsilon=1e-6, step=0.01, is_draw=True):
    x1 = symbols('x1')
    y1 = symbols('y1')
    y2 = symbols('y2')
    my_f = []
    my_f.append(y2)
    my_f.append(f)
    (a, b) = C
    if type==1:
        my_C = (a, b+step)
        (z0, z1) = initial_z
        (alpha, beta) = initial_point
        for i in range(iteration):
            my_initial_point0 = np.array([[a, alpha],[a, z0]])
            my_initial_point1 = np.array([[a, alpha],[a, z1]])
            _, temp_y = differential_equations_runge_kutte4(my_f, my_C, my_initial_point0, step=step, is_draw=False)
            y0 = temp_y[0, temp_y.shape[1]-1]
            _, temp_y = differential_equations_runge_kutte4(my_f, my_C, my_initial_point1, step=step, is_draw=False)
            y1 = temp_y[0, temp_y.shape[1] - 1]
            z = z1 - (y1 - beta) / (y1 - y0) * (z1 - z0)
            if abs(y1-beta)<epsilon:
                break
            z0 = z1
            z1 = z
        my_initial_point = np.array([[a, alpha], [a, z]])
        X, Y = differential_equations_runge_kutte4(my_f, my_C, my_initial_point, step=step, is_draw=is_draw)
    elif type==2:
        my_C = (a, b + step)
        (z0, z1) = initial_z
        (alpha, beta) = initial_point
        for i in range(iteration):
            my_initial_point0 = np.array([[a, z0], [a, alpha]])
            my_initial_point1 = np.array([[a, z1], [a, alpha]])
            _, temp_y = differential_equations_runge_kutte4(my_f, my_C, my_initial_point0, step=step, is_draw=False)
            yp0 = temp_y[1, temp_y.shape[1]-1]
            _, temp_y = differential_equations_runge_kutte4(my_f, my_C, my_initial_point1, step=step, is_draw=False)
            yp1 = temp_y[1, temp_y.shape[1] - 1]
            z = z1 - (yp1 - beta) / (yp1 - yp0) * (z1 - z0)
            if abs(yp1-beta)<epsilon:
                break
            z0 = z1
            z1 = z
        my_initial_point = np.array([[a, z], [a, alpha]])
        X, Y = differential_equations_runge_kutte4(my_f, my_C, my_initial_point, step=step, is_draw=is_draw)
    elif type==3:
        my_C = (a, b + step)
        (z0, z1) = initial_z
        (alpha0, alpha1, beta0, beta1) = initial_point
        for i in range(iteration):
            my_initial_point0 = np.array([[a, z0], [a, (z0-alpha1)/alpha0]])
            my_initial_point1 = np.array([[a, z1], [a, (z1-alpha1)/alpha0]])
            _, temp_y = differential_equations_runge_kutte4(my_f, my_C, my_initial_point0, step=step, is_draw=False)
            y0 = temp_y[0, temp_y.shape[1]-1]
            yp0 = temp_y[1, temp_y.shape[1]-1]
            _, temp_y = differential_equations_runge_kutte4(my_f, my_C, my_initial_point1, step=step, is_draw=False)
            y1 = temp_y[0, temp_y.shape[1]-1]
            yp1 = temp_y[1, temp_y.shape[1] - 1]
            z = z1 - (y1-beta0*yp1 - beta1) / (y1-beta0*yp1 - y0-beta0*yp0) * (z1 - z0)
            if abs(y1-beta0*yp1 - beta1)<epsilon:
                break
            z0 = z1
            z1 = z
        my_initial_point = np.array([[a, z], [a, (z-alpha1)/alpha0]])
        X, Y = differential_equations_runge_kutte4(my_f, my_C, my_initial_point, step=step, is_draw=is_draw)
    else:
        print('type only admit 1,2,3!')
        return
    return X, Y

# 10.有限差分法 y''=p(x)*y'+q(x)*y+r(x)
def finite_difference_method(p, q, r, C, initial_point, num_section=10, type=1, is_draw=True):
    x = symbols('x')
    (a, b) = C
    h = (b-a)/num_section
    X = np.linspace(a, b, num_section+1)
    A = np.zeros((num_section+1, num_section+1))
    B = np.zeros((num_section+1, 1))
    if type==1:
        (alpha, beta) = initial_point
        for i in range(1, num_section):
            A[i, i-1] = 1+h*p.subs({x: X[i]})/2
            A[i, i] = -2-h**2*q.subs({x: X[i]})
            A[i, i+1] = 1-h*p.subs({x: X[i]})/2
            B[i, 0] = h**2*r.subs({x: X[i]})
        A[0, 0] = 1
        A[num_section, num_section] = 1
        B[0, 0] = alpha
        B[num_section, 0] = beta
    elif type==2:
        (alpha, beta) = initial_point
        for i in range(1, num_section):
            A[i, i-1] = 1+h*p.subs({x: X[i]})/2
            A[i, i] = -2-h**2*q.subs({x: X[i]})
            A[i, i+1] = 1-h*p.subs({x: X[i]})/2
            B[i, 0] = h**2*r.subs({x: X[i]})
        A[0, 0] = -1
        A[0, 1] = 1
        A[num_section, num_section-1] = -1
        A[num_section, num_section] = 1
        B[0, 0] = h*alpha
        B[num_section, 0] = h*beta
    elif type==3:
        (alpha0, alpha1, beta0, beta1) = initial_point
        for i in range(1, num_section):
            A[i, i-1] = 1+h*p.subs({x: X[i]})/2
            A[i, i] = -2-h**2*q.subs({x: X[i]})
            A[i, i+1] = 1-h*p.subs({x: X[i]})/2
            B[i, 0] = h**2*r.subs({x: X[i]})
        A[0, 0] = alpha0+h
        A[0, 1] = -alpha0
        A[num_section, num_section-1] = beta0
        A[num_section, num_section] = h-beta0
        B[0, 0] = alpha1*h
        B[num_section, 0] = beta1*h
    else:
        print('type only admit 1,2,3!')
        return
    Y = crout_decomposition(A, B)
    if is_draw == True:
        plt.plot(X, Y, 'r', linewidth=2)
        plt.show()
    return X, Y
