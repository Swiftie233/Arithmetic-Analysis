"""
使用二分法、牛顿法、割线法、修正牛顿法和拟牛顿法求解非线性方程（组）
@author:Swiftie233
@Date:2020/10/20
@Harbin Institute of Technology
"""

from math import sin, exp
import sympy as sp
import numpy as np
import copy as cp


def diff(f, x):
    return f.diff(x)


def bisection_method():
    def f1(x): return sin(x)-0.5*x**2
    a = [1]
    b = [2]
    epsilon = 0.5*10**-5
    x = []
    i = 0
    while(1):
        x.append((a[i]+b[i])/2)
        if f1(x[i])*f1(a[i]) < 0:
            a.append(a[i])
            b.append(x[i])
        elif f1(x[i])*f1(b[i]) < 0:
            a.append(x[i])
            b.append(b[i])
        if (b[i]-a[i]) < epsilon:
            x.append((a[i]+b[i])/2)
            break
        i += 1
    return x


def newton(f, x, n):
    t = sp.symbols('t')
    fx = sp.diff(f)
    for i in range(n):
        x.append(x[i]-f.evalf(subs={t: x[i]})/fx.evalf(subs={t: x[i]}))
    print(x[-1])
    # return x


def secant_method(f, x, n):
    # 割线法
    denominator = []
    numerator = []
    try:
        for i in range(1, n):
            denominator.append(f(x[i])-f(x[i-1]))
            numerator.append(f(x[i])*(x[i]-x[i-1]))
            x.append(x[i]-numerator[i-1]/denominator[i-1])
    except:
        pass
    print(x[-1])


def mended_newton(f, x, n):
    t = sp.symbols('t')
    fx = sp.diff(f)
    for i in range(n):
        x.append(x[i]-2*f.evalf(subs={t: x[i]})/fx.evalf(subs={t: x[i]}))
    print(x[-1])


def quasi_newton(F, t, H, n):
    """
    F:传入原方程组，可以计算Fx
    t:方程的迭代解，为3xN矩阵，每一列为迭代结果
    H:传入的H0
    n:预定的迭代次数
    """
    x, y, z = sp.symbols('x y z')

    FF = cp.deepcopy(F)
    r = np.array([[], [], []])
    yy = np.array([[], [], []])
    ans = np.array(FF.subs([(x, t[0][0]), (y, t[1][0]), (z, t[2][0])]).evalf())

    for i in range(1, n):
        h = np.split(H, i, 1)
        temp_t = t[:, i-1]-h[i-1]@ans[:, i-1]  # xi+1
        t = np.c_[t, temp_t]

        ans = np.c_[ans, F.subs({x: t[0][i], y:t[1][i], z:t[2][i]}).evalf()]

        temp_r = t[:, i]-t[:, i-1]  # ri
        r = np.c_[r, temp_r]

        yy = np.c_[yy, ans[:, i]-ans[:, i-1]]

        t1=(r[:,i-1]-h[i-1]@yy[:,i-1]).reshape([3,1])
        t2=(r[:,i-1]@h[i-1]).reshape([1,3])
        t3=r[:,i-1]@h[i-1]@yy[:,i-1]

        H = np.c_[H, h[i-1]+t1@t2/t3]

    print(t[:,-1])


def call_bisection():
    x = bisection_method()
    print("二分法求解 sin(x)-0.5*x**2")
    print(x[-1])
    # print(a)
    # print(b)
    # print(i)
    # return x,a,b,i


def call_newton():
    t = sp.symbols('t')
    f1 = t*sp.exp(t)-1
    f2 = t**3-3-1
    f3 = (t-1)**2*(2*t-1)
    n = eval(input('请输入迭代次数：'))

    print('牛顿迭代法求解 t*exp(t)-1，初值为0.5')
    newton(f1, [0.5], n)

    print('牛顿迭代法求解 t**3-3-1，初值为1')
    newton(f2, [1], n)

    print('牛顿迭代法求解 (t-1)**2*(2*t-1)，初值为0.45')
    newton(f3, [0.45], n)

    print('牛顿迭代法求解 (t-1)**2*(2*t-1)，初值为0.65')
    newton(f3, [0.65], n)


def call_secant_method():
    def f(x): return x*exp(x)-1
    x = [0.4, 0.6]
    n = eval(input('输入迭代次数：'))
    print('割线法求解 x*exp(x)-1，初值为0.4，0.6')
    secant_method(f, x, n)


def call_mended_newton():
    t = sp.symbols('t')
    f = (t-1)**2*(2*t-1)
    n = eval(input('请输入迭代次数：'))

    print('牛顿迭代法求解 (t-1)**2*(2*t-1)，初值为0.55')
    mended_newton(f, [0.55], n)


def call_quasi_newton():
    x, y, z = sp.symbols('x y z')

    f1 = x*y-z**2-1
    f2 = x*y*z+y**2-x**2-2
    f3 = sp.exp(x)+z-sp.exp(y)-3

    t = np.array([[1, ], [1, ], [1, ]])
    F = sp.Matrix([[f1], [f2], [f3]])
    v = sp.Matrix([x, y, z])
    Fx = F.jacobian(v)
    H = np.array(
        Fx.subs({x: t[0][0], y: t[1][0], z: t[2][0]}).evalf())  # 返回矩阵H0
    H = H.astype(np.float)
    H = np.linalg.inv(H)
    # n=eval(input('输入迭代次数：'))
    n = 10
    quasi_newton(F, t, H, n)


def main():
    call_bisection()
    call_newton()
    call_secant_method()
    call_mended_newton()
    call_quasi_newton()


main()
