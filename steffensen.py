"""实现Steffensen加速收敛方法，输出结果保留4位小数"""
"""算法原理参照《数值分析原理》吴勃英版p31"""
#author:Swiftie233
#date:2020/10/1

def input_part():
    string=input("输入f(x)=0的表达式：")
    x0=eval(input("输入迭代初值x0："))
    n=eval(input("输入迭代步长n："))
    y=lambda x: eval(string)
    return string,x0,n,y

def func(x1,x2,x3):
    return (x1*x3-x2**2)/1.0/(x3-2*x2+x1)

def iteration1(string,x0,n,y):
    x=[x0]
    xx=[0]
    xxx=[0]
    
    for i in range(1,n):
        xx.append(y(x[i-1]))
        xxx.append(y(xx[i]))
        x.append(func(x[i-1],xx[i],xxx[i]))
       

    return x
[string,x0,n,y]=input_part()
x=iteration1(string,x0,n,y)
for _ in range(len(x)):
    print('x为：%.4f'%x[_])
