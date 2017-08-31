# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:27:38 2017

@author: Aadila
"""

import numpy as np

#import astropy.io
from astropy.io import fits
import matplotlib.pyplot as plt


ncomp=1
nfreq=2
xmin=3000
xmax=7000
ymin=200
ymax=1800

act150=fits.getdata('weightedMap_4.fits')[ymin:ymax,xmin:xmax]/1000000
pl220=fits.getdata('HFI_equD56_217.fits')[ymin:ymax,xmin:xmax]

#diff=act150-pl220

m=ymax-ymin
n=xmax-xmin

x=np.arange(n); y=np.arange(m)  
x[int(n/2):]=x[int(n/2):]-n; y[int(m/2):]=y[int(m/2):]-m
xx,yy=np.meshgrid(x,y)


rr=xx**2+yy**2
del xx
del yy

def fwhm2sigma(fwhm):
	return fwhm/np.sqrt(8*np.log(2))  
sigma=fwhm2sigma(4)

temp=np.exp(-rr/(2*sigma**2))

del rr,sigma
fttemp=np.fft.fft2(temp)

del temp
ft150=(np.fft.fft2(act150))
ft220=(np.fft.fft2(pl220))

#del act150
#pl220a


bigkernel= np.zeros((m,n))
kernel=np.array([[0.5,0.7,0.5],[0.7,1,0.7],[0.5,0.7,0.5]])
kernel=kernel/kernel.sum()

bigkernel[:3,:3]=kernel
bigkernel=np.roll(np.roll(bigkernel,-1,0),-1,1)
ftbigkernel=np.fft.fft2(bigkernel)



#Making the noise matrix

a=ft150*np.conj(ft150)
b=ft150*np.conj(ft220)
#c=np.conj(b)
d=ft220*np.conj(ft220)

a=np.real(np.fft.ifft2(np.fft.fft2(a)*(ftbigkernel)))
b=np.real(np.fft.ifft2(np.fft.fft2(b)*(ftbigkernel))) # physics reasons 
d=np.real(np.fft.ifft2(np.fft.fft2(d)*(ftbigkernel)))





#del n150
#del n220

ninv_a=np.zeros((2,m*n))
ninv_a[0,:]=np.ravel(np.divide(fttemp*d-0*b, a*d-b*b))
ninv_a[1,:]=np.ravel(np.divide(-fttemp*b+0*a, a*d-b*b))


#ordering ninv_a as we had ordered A previously .      
ninv_a=np.reshape(ninv_a, (2*m*n,ncomp),order='C')

A=np.zeros((nfreq*m*n,ncomp))
A[:m*n,:]=np.reshape(fttemp,(m*n,1))
A[m*n:,:]=0


#lhs=A.transpose()*ninv_a #- gives error:iterator is too large 

lhs=np.sum(A*ninv_a) 

ninv_d1=np.divide(ft150*d-ft220*b,a*d-b*b)
ninv_d2=np.divide(-ft150*b+ft220*a,a*d-b*b)

del a,b,d, # ft150, ft220

rhs=np.real(np.fft.ifft2(fttemp*(ninv_d1)+0*(ninv_d2)))
rhs=np.ravel(rhs)

fit=(np.divide(rhs,lhs))

cluster=np.real(fit.reshape(m,n))

#var=(np.square(np.std(cluster)))

#index= np.argwhere(cluster<5*var)
#print (index)

plt.imshow(cluster), plt.colorbar()



#This shows that the smoothing works 

# plt.imshow(np.real(act150), vmin=-10e-10, vmax=10e-10),
# plt.figure(),
# plt.imshow(np.real(np.fft.ifft2(ft150*ftbigkernel)), vmin=-10e-10, vmax=10e-10)