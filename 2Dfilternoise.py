# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 10:52:39 2017

@author: Aadila
"""


import numpy as np
import matplotlib.pyplot as plt


n=1000
m=1000
sigma=20



x=np.arange(n); y=np.arange(m)  
x[n/2:]=x[n/2:]-n; y[m/2:]=y[m/2:]-m

  
xx,yy=np.meshgrid(x,y)

rr=xx**2+yy**2

temp=np.exp(-rr/(2*sigma**2))
fttemp=np.fft.fft2(temp)

fttempsq=np.sum(fttemp**2)

d150=10*np.random.randn(m,n) 
d220=10*np.random.randn(m,n)
d220=d220+100*np.roll(np.roll(temp,300,axis=0),200,axis=1)
ncorr=5*np.random.randn(m,n)

d150=d150+ncorr
d220=d220+ncorr  
ftd150=np.fft.fft2(d150)
ftd220=np.fft.fft2(d220) 
                         
                         
#Noise estimates

noise=(np.matrix([[125,25],[25,125]]))
ninv=np.linalg.inv(noise)



array=np.array([[ninv[0,0]+ninv[1,0]+ninv[0,1]+ninv[1,1],ninv[0,0]+ninv[1,0]],[ninv[0,0]+ninv[0,1],ninv[0,0]]])
lhs=(fttempsq*array)
#lhs=np.fft.ifft2(fttempsq*array)


rhscmb=(fttemp*(ftd150*(ninv[0,0]+ninv[1,0])+ftd220*(ninv[0,1]+ninv[1,1])))
rhscluster=(fttemp*(ftd150*ninv[0,0]+ftd220*ninv[0,1]))


rhscmb=np.fft.ifft2(fttemp*(ftd150*(ninv[0,0]+ninv[1,0])+ftd220*(ninv[0,1]+ninv[1,1])))
rhscluster=np.fft.ifft2(fttemp*(ftd150*ninv[0,0]+ftd220*ninv[0,1]))


rhs=np.zeros((2,m*n))
rhs[0,:]=np.ravel(rhscmb)
rhs[1,:]=np.ravel(rhscluster)
#
fit=(np.matmul(np.linalg.inv(lhs),rhs))
cmb=np.real(fit[0,:].reshape(m,n))
cluster=np.real(fit[1,:].reshape(m,n))
#
#
plt.imshow(cmb), plt.colorbar(),plt.title('CMB')
plt.figure()
plt.imshow(cluster), plt.colorbar(), plt.title("cluster")


plt.show()

#
                    