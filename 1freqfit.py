# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 18:08:07 2017

@author: Aadila


"""

import numpy as np
import matplotlib.pyplot as plt 

n=100
sigma1=0.5
sigma2=0.6
x=np.linspace(-5,5,n)
data=np.random.randn(n)



A1=np.exp(-x**2/(2*sigma1**2))
A2=np.exp(-x**2/(2*sigma2**2))

A=np.zeros(2*n)
A[:n]=A1
A[n:]=A2

 
fta=np.matrix(np.fft.fft(A))
lhs=fta.transpose()*fta
ftdata=np.matrix(np.fft.fft(data))
rhs=fta.transpose()*ftdata

ftpred=fta*np.linalg.inv(lhs)*rhs

pred=np.real(np.fft.ifft(ftpred))

plt.ion()                      
plt.plot(x, data, 'r')
plt.plot(x, pred.transpose(), 'b')
