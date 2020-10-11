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
print(A)

