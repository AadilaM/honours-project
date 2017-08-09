# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:55:31 2017

@author: Aadila
"""

import numpy as np
import matplotlib.pyplot as plt 

n=100
start=-5
end=5
xrange=end-start
sigma=0.5
x=np.linspace(start,end,n)

 
temp=np.exp(-x**2/(2*sigma**2))

d150=np.random.randn(n) 
d220=np.random.randn(n)

 

amp=10
d150=d150+amp*np.roll(temp,40)
#d220=d220+amp*temp
#               
A=np.zeros((2*n,2))
#
#
A[0:n,0]=temp #CMB 150
A[n:,0]= temp #CMB 220
A[0:n,1]= temp #CP 150
A[n:,1]= 0*temp #CP 220

aa=np.matrix(A)
lhs=aa.transpose()*aa
                
ftd150=np.fft.fft(d150)
ftd220=np.fft.fft(d220)

rhs=np.zeros((2,n))
rhs[0,:]=np.real(np.fft.ifft(np.fft.fft(temp)*ftd150+np.fft.fft(temp)*ftd220))

rhs[1,:]=np.real(np.fft.ifft(np.fft.fft(temp)*ftd150))
    
fitp=np.linalg.inv(lhs)*rhs


#Plotting the predicted models 
#pred=aa*np.linalg.inv(lhs)*rhs              
##
##for i in range(0,n):
##    column=pred[:,i]
##    pred150=column[0:n]
##    pred220=column[n:]
##    plt.plot(x,np.roll(pred150,int(x[i]*n/10)) ) #10 comes from n/interval
##    plt.plot(x,np.roll(pred220,int(x[i]*n/10)) )

fitpcmb=np.ravel(fitp[0,:])
fitpcluster=np.ravel(fitp[1,:])


plt.plot(fitpcmb,'b'), plt.plot(fitpcluster,'k')

plt.show()
    
