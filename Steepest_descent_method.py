"""最速下降法python实现"""
import numpy as np

def init():
  x=[]
  while(1):
    try:
      t=eval(input('Input the initial number of x:'))
      x.append([t])
    except:
      break
  x=np.array(x)

  A=[]
  while(1):
    tt=[]
    
    while(1):
      
      try:
        t=eval(input('Input the initial number of A:'))
        tt.append(t)
      except:
        break
      
    if input()=='':
      break
    else:
      A.append(tt)

  A=np.array(A)
A=[[2,1],[1,5]]
A=np.array(A)
b=np.array([[3],[1]])

nums=eval(input("input the length of x"))
#x=np.empty([nums,1],dtype=float,order='C') #定义一x空数组
x=np.array([0,0],dtype=float)
x=x.T
r=b-A@x
alpha=inner(r,r)/inner(np.dot(A,r),r)

#r=np.empty([nums,1],dtype=float,order='C') #定义一r空数组

#np.c_(a,b) 在矩阵a右侧插入矩阵b
#np.inner(a,b)计算向量a，b的内积

def inner(a,b):
  return np.inner(a,b)
def fun(x,r,alpha,A,b):
  i=0
  
  while(1):
    np.c_(r,b-A@x[i])
    np.c_(alpha,np.inner(r[i],r[i])/np.inner(np.dot(A,r[i]),r[i]))
    np.c_(x,x[i]+alpha[i]*r[i])
    i+=1
    if i>10:
      break
  return x

print(A)
print(b)