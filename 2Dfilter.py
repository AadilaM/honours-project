# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 16:00:11 2017

@author: Aadila
"""

import numpy as np

import matplotlib.pyplot as plt

n=1000
m=1000
sigma=20



x=np.arange(n); y=np.arange(m) #with wraparound 
x[n/2:]=x[n/2:]-n; y[m/2:]=y[m/2:]-m

  
xx,yy=np.meshgrid(x,y)

rr=xx**2+yy**2

temp=np.exp(-rr/(2*sigma**2))

d150=np.random.randn(m,n) 
d220=np.random.randn(m,n)

#d150=d150+101*np.roll(np.roll(temp,30,axis=0),20,axis=1)
d220=d220+10*np.roll(np.roll(temp,300,axis=0),200,axis=1)


#Now we make A 

tempsq=np.sum(temp*temp)

array=np.array([[2,1],[1,1]])
lhs=tempsq*array

      

rhscmb=np.real(np.fft.ifft2(np.fft.fft2(temp)*np.fft.fft2(d150)+np.fft.fft2(temp)*np.fft.fft2(d220)))
rhscluster=np.real(np.fft.ifft2(np.fft.fft2(temp)*np.fft.fft2(d150)))



rhs=np.array([[np.ravel(rhscmb)],[np.ravel(rhscluster)]]).reshape(2,m*n)

fit=np.matmul(np.linalg.inv(lhs),rhs)
cmb=fit[0,:].reshape(m,n)
cluster=fit[1,:].reshape(m,n)


plt.imshow(cmb), plt.colorbar(),plt.title('CMB')
plt.show()
plt.imshow(cluster), plt.colorbar(), plt.title("cluster")
plt.show()
                  




