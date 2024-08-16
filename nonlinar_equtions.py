import numpy as np
from sympy import *
from numerical_analysis.supplement import compute_Jocab

# 1.二分法
def dichotomy(func, x_initial, iteration=20, accuracy=0.00001):
    x0, x1 = x_initial
    i = 0
    root = x0
    while i < iteration:
        y0 = func(x0)
        root = (x0 + x1) / 2
        print(str((x0+x1)/2))
        y_middle = func((x0+x1)/2)
        if abs(y_middle) < accuracy:
            break
        else:
            if y_middle*y0 < 0:
                x1 = (x0+x1)/2
            else:
                x0 = (x0+x1)/2
        i += 1
    return root

# 2.Newton迭代法 (已知根的重数，简化Newton迭代法，Newton下山法)
# r:根的重数  n:截断长度
def root_Newton(f, x_initial, r=1, iteration=10, accuracy=0.00001, n=8, simplify=False,
                      Downhill_method=False, lambd=0.5):
    root = x_initial
    x = symbols('x')
    func = f
    derivative = diff(f, x)
    modified_derivatives = derivative.subs({x: root}).evalf(n)
    i = 0
    while i < iteration:
        y = func.subs({x: root}).evalf(n)
        print(str(root))
        if abs(y) < accuracy:
            break
        elif simplify:
            root = root - r*y/modified_derivatives
        elif Downhill_method:
            root = root - r*lambd*y/derivative.subs({x: root}).evalf(n)
        else:
            root = root - r*y/derivative.subs({x: root}).evalf(n)
        i += 1
    return root

# 3.Newton迭代法 (根的重数未知)
def unknow_root_Newton(f, x_initial, iteration=10, accuracy=0.00001, n=8):
    root = x_initial
    x = symbols('x')
    u = f/diff(f, x)
    u_derivative = diff(u, x)
    i = 0
    while i < iteration:
        y = f.subs({x: root}).evalf(n)
        print(str(root))
        if abs(y) < accuracy:
            break
        else:
            root = root - u.subs({x: root}).evalf(n)/u_derivative.subs({x: root}).evalf(n)
        i += 1
    return root

# 4.多点迭代法 (割线法、虚位法)
def root_Secant(f, x_initial, iteration=20, accuracy=0.00001, n=16):
    x1 = np.linspace(1, 3*iteration)
    i = 0
    x1[i*3] = x_initial
    x = symbols('x')
    x1[i*3+1] = x1[i*3] - f.subs({x: x_initial}).evalf(n)/diff(f, x).subs({x: x_initial}).evalf(n)
    x1[i*3+2] = (x1[i*3]+x1[i*3+1])/2
    while i < iteration:
        i += 1
        y1 = f.subs({x: x1[(i-1)*3]}).evalf(n)
        y2 = f.subs({x: x1[(i-1)*3+1]}).evalf(n)
        y3 = f.subs({x: x1[(i-1)*3+2]}).evalf(n)
        root =x1[(i-1)*3+2]
        print(str(root))
        if abs(y3) < accuracy:
            break
        if y1*y3 > 0:
            x1[i*3] = x1[(i-1)*3+1]
            x1[i*3+1] = x1[(i-1)*3+2]
            x1[i*3+2] = y3/(y3-y2)*x1[(i-1)*3+1]+y2/(y2-y3)*x1[(i-1)*3+2]
        else:
            x1[i * 3] = x1[(i - 1) * 3 ]
            x1[i * 3 + 1] = x1[(i - 1) * 3 + 2]
            x1[i * 3 + 2] = y3 / (y3 - y2) * x1[(i - 1) * 3 + 1] + y2 / (y2 - y3) * x1[(i - 1) * 3 + 2]
    return root

# 5.Newton迭代法解非线性方程组
# f为list,x_initial为list
def roots_Newton(f, x_initial, iteration=10, accuracy=0.00001, n=16):
    m = len(f)
    x = symbols('x1:%d' % (m + 1)) #定义多个变量
    A = compute_Jocab(f, x_initial, n)
    root = x_initial
    root_v = np.zeros((m, 1))
    for j in range(m):
        root_v[j, 0] = root[j]
    y = np.zeros((m, 1))
    i = 0
    while i < iteration:
        root_q = []
        for j in range(m):
            root_q.append(root_v[j, 0])
        for j in range(m):
            y[j, 0] = f[j].subs(dict(zip(x, root_q))).evalf(n) #x和root_q均为list
        if ((abs(y)) < accuracy).all():
            break
        else:
            root_v = root_v - np.linalg.inv(A)*y
        print(str(root_q))
        i += 1
    return root_q

# 6.Broyden秩1方法
def roots_Broyden(f, x_initial, iteration=20, accuracy=0.00001, n=16):
    m = len(f)
    x = symbols('x1:%d' % (m + 1))
    root = x_initial
    root_v = np.zeros((m, 1))
    for j in range(m):
        root_v[j, 0] = root[j]
    y = np.zeros((m, 1))
    i = 0
    A = compute_Jocab(f, x_initial, n)
    while i < iteration:
        root_q = []
        for j in range(m):
            root_q.append(root_v[j, 0])
        for j in range(m):
            y[j, 0] = f[j].subs(dict(zip(x, root_q))).evalf(n)
        if ((abs(y)) < accuracy).all():
            break
        else:
            r = -np.linalg.inv(A) * y
            root_v = root_v - np.linalg.inv(A) * y
            A = A + (y-A*r)*r.T/(r.T*r)
        print(str(root_q))
        i += 1
    return root_q

# 7.Broyden秩1方法(逆)
def roots_Broyden_inv(f, x_initial, iteration=20, accuracy=0.00001, n=16):
    m = len(f)
    x = symbols('x1:%d' % (m + 1))
    root = x_initial
    root_v = np.zeros((m, 1))
    for j in range(m):
        root_v[j, 0] = root[j]
    y = np.zeros((m, 1))
    i = 0
    A = compute_Jocab(f, x_initial, n)
    H = np.linalg.inv(A)
    while i < iteration:
        root_q = []
        for j in range(m):
            root_q.append(root_v[j, 0])
        for j in range(m):
            y[j, 0] = f[j].subs(dict(zip(x, root_q))).evalf(n)
        if ((abs(y)) < accuracy).all():
            break
        else:
            r = -H*y
            root_v = root_v - H * y
            H = H + (r-H*y)*((r.T*H)/(r.T*H*y))
        print(str(root_q))
        i += 1
    return root_q


