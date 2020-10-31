"""Romberg Integration"""
import math as mt
import numpy as np


def init():

    a = eval(input("输入区间左端点值a："))
    b = eval(input("输入区间右端点值b："))
    k = eval(input("输入龙贝格阶数2**k："))
    ff = input("输入被积函数f(x)：")
    epsilon = eval(input("输入精度值："))
    def f(x): return eval(ff)
    """ 调试数据
    a = 6
    b = 100
    k = 5
    epsilon = 0.5*10**-1
    def f(x): return x**3 """

    T = np.full([k+1, k+1], 10**-20)
    T[0][0] = (b-a)/2*(f(a)+f(b))

    return a, b, k, f, T, epsilon


def accelarate(T, k, eps):
    """
    Richardson加速法
    """
    for m in range(1, k+1):
        flag = 1
        for kk in range(0, k+1-m):
            T[kk][m] = (4**m*T[kk+1][m-1]-T[kk][m-1])/(4**m-1)

            ess = abs(T[0][m]-T[0][m-1])

            if ess < eps:
                flag = 0
                break
        if flag == 0:
            break

    return T


def recurrence(T, k, f, a, b):
    """
    递推
    """

    for kk in range(0, k):
        sum = 0
        n = 2**kk
        h = (b-a)/n
        for i in range(n):
            sum += f(a+h*(i+0.5))

        T[kk+1][0] = 0.5*T[kk][0]+0.5*h*sum

    return T


def post_processing(T):
    [m, n] = T.shape
    eps = 10**-20

    for i in range(m):
        if T[0][i] == eps:
            T = np.split(T, [i], axis=1)[0]
            break

    return T

def main():

    [a, b, k, f, T, epsilon] = init()
    T = recurrence(T, k, f, a, b)
    T = accelarate(T, k, epsilon)
    T = post_processing(T)
    print(T)
    print("结果是：{}".format(T[0][-1]))
    print("Epsilon = {}".format(abs(T[0][-1]-T[0][-2])))


main()
