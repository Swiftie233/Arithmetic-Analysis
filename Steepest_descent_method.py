"""最速下降法python实现"""
#arthor:Swiftie233
#date:2020/10/11

import numpy as np
def getx():
  x=[]
  while(1):
    try:
      t=eval(input('Input the initial number of x:'))
      x.append([t])
    except:
      break
  x=np.array(x)

def matrix_get():
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


  

def alpha_calc(r,A,i):
  #i为r的第i列
  return np.inner(r[:,i],r[:,i])/np.inner(np.dot(A,r[:,i]),r[:,i])

def init():
  A=[[2,1],[1,5]]
  b=[[3],[1]]
  x=[[0],[0]]
  
  A=np.array(A,dtype=float)
  b=np.array(b,dtype=float)
  x=np.array(x,dtype=float)
 
  r=b-A@x
  alpha=[alpha_calc(r,A,0)]

  return A,b,x,r,alpha

def fun(A,b,x,r,alpha):
  i=0
  
  while(1):
    x=np.c_[x,x[:,i]+alpha[i]*r[:,i]]
    r=np.c_[r,b-A@x[:,i+1].reshape(2,1)]
    alpha.append(alpha_calc(r,A,i+1))
    i+=1
    if i>20:
      break
  return x,r,alpha

def main():
  [A,b,x,r,alpha]=init()
  [x,r,alpha]=fun(A,b,x,r,alpha)
  print('x的值为：')
  print(x[:,10:])
  print('r的值为：')
  print(r[:,10:])
  print('alpha的值为：')
  print(alpha[10:])
  print(x.shape)
  

main()