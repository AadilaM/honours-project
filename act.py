# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:26:24 2017

@author: Aadila
"""

import numpy as np
import pyfits 
import matplotlib.pyplot as plt

xmin=6000
xmax=12000


map150=pyfits.getdata("ACT_148_south_season_2_1way_v3_src_free.fits")[:,xmin:xmax]
map220=pyfits.getdata("ACT_220_south_season_2_1way_v3_srcfree.fits")[:1177,xmin:xmax]
hits150=pyfits.getdata("ACT_148_south_season_2_1way_hits_v3.fits")[:,xmin:xmax]
hits220=pyfits.getdata("ACT_220_south_season_2_1way_hits_v3.fits")[:1177,xmin:xmax]

ftmap150=np.fft.fft2(map150*hits150)
ftmap220=np.fft.fft2(map220*hits220)

del map150
del map220
del hits150
del hits220


m=1177
n=xmax-xmin

def fwhm2sigma(fwhm):
	return fwhm/np.sqrt(8*np.log(2))  
sigma=fwhm2sigma(8)


x=np.arange(n); y=np.arange(m)  
x[n/2:]=x[n/2:]-n; y[m/2:]=y[m/2:]-m
xx,yy=np.meshgrid(x,y)


rr=xx**2+yy**2
del xx
del yy


temp=np.exp(-rr/(2*sigma**2))

del rr
fttemp=np.fft.fft2(temp)

tempsq=np.sum(temp*temp)

del temp
array=np.array([[2,1],[1,1]])
lhs=tempsq*array

     
rhscmb=np.real(np.fft.ifft2(fttemp*ftmap150+fttemp*ftmap220))
rhscluster=np.real(np.fft.ifft2(fttemp*ftmap150))

rhs=np.array([[np.ravel(rhscmb)],[np.ravel(rhscluster)]]).reshape(2,m*n)

del rhscmb
del rhscluster

fit=np.matmul(np.linalg.inv(lhs),rhs)
cmb=fit[0,:].reshape(m,n)
cluster=fit[1,:].reshape(m,n)


#plt.imshow(cmb), plt.colorbar(),plt.title('CMB')
#plt.figure()
#plt.imshow(cluster), plt.colorbar(), plt.title("cluster")
#plt.show()
             