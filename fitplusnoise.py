# -*- coding: utf-8 -*-
"""
2D filter with noise 
same noise matrix (np.random.randn) for each pixel 
2 frequencies, 2 components
                  
@author: Aadila
"""

#%%
import numpy as np
import matplotlib.pyplot as plt


n=1000
m=1000
sigma=20



x=np.arange(n); y=np.arange(m)  
x[int(n/2):]=x[int(n/2):]-n; y[int(m/2):]=y[int(m/2):]-m

  
xx,yy=np.meshgrid(x,y)

rr=xx**2+yy**2

temp=np.exp(-rr/(2*sigma**2))

fttemp=np.fft.fft2(temp)



d150=10*np.random.randn(m,n) 
d220=10*np.random.randn(m,n)
d220=d220+100*np.roll(np.roll(temp,300,axis=0),200,axis=1)
ncorr=5*np.random.randn(m,n)

d150=d150+ncorr
d220=d220+ncorr  

ftd150=np.fft.fft((d150))
ftd220=np.fft.fft((d220)) 
                         
                         
#Noise estimates

noise=(np.matrix([[125,25],[25,125]]))
ninv=np.linalg.inv(noise)


# NB: ravel uses 'c' ordering by default

cmb=np.zeros((2,m*n))
cmb[0,:]=np.ravel(temp) #cmb150 
cmb[1,:]=np.ravel(temp) #cmb220
ninv_cmb=np.matmul(ninv,cmb )

ninv_cmb=np.ravel(ninv_cmb,order='F') #just use ravel?

cluster=np.zeros((2,m*n))
cluster[0,:]=np.ravel(temp) #cluster150
cluster[1,:]=0 #cluster220
ninv_cl=np.matmul(ninv,cluster )

ninv_cl=np.ravel(ninv_cl,order='F')

ninv_a=np.zeros((2*m*n,2))
ninv_a[:,0]=(ninv_cmb)
ninv_a[:,1]=(ninv_cl)





a=np.zeros((2*m*n,2))
a[:m*n,0]=np.ravel(temp)
a[:m*n,1]=np.ravel(temp)
a[m*n:,0]=np.ravel(temp) 
a[m*n:,1]=0

a=np.matrix(a)
ninv_a=np.matrix(ninv_a)
lhs=a.transpose()*ninv_a

              


ninv_d150=ninv[0,0]*d150+ninv[0,1]*d220
ninv_d220=ninv[1,0]*d150+ninv[1,1]*d220              

rhscmb=np.real(np.fft.ifft2(np.fft.fft2(ninv_d150)*fttemp+np.fft.fft2(ninv_d220)*fttemp))
rhscluster=np.real(np.fft.ifft2(np.fft.fft2(ninv_d150)*fttemp))              
              
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