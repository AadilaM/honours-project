# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 11:21:42 2017

@author: Aadila
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.io import fits

xmin=3000
xmax=7000
ymin=200
ymax=1800
thresh=4
filename='ap2to5b.txt'

    
def get_ang_radius (comovingsize=8, z=0.5, angular_diameter_distance=1272.5) : #in degrees, comoving in MPC
    phys_size=comovingsize/(1+z)
    return phys_size*180/(angular_diameter_distance*np.pi*2)#gives radius not diameter

 
def open_profile(filename):#ensure filename is a string
    PIX = 2048 # all files are 2048 x 2048
    fd = open(filename, 'rb')
    temp = np.fromfile(file=fd, dtype=np.float32)
    fd.close()
    profile = np.reshape(temp,(PIX,PIX))
    temp = 0
    #zoom = PIX/2 -128 - if we want the zoomed in cluster profile
    #profile=profile[int(zoom):-int(zoom),int(zoom):-int(zoom)] 
    return profile 

def get_r_matrix(data,rmax): #gives centred distance matrix in pixels 
    centre_y= int((data.shape[0])/2)
    centre_x= int((data.shape[1])/2)
    y, x = np.indices((data.shape))
    r = np.sqrt((x - centre_x)**2 + (y - centre_y)**2)
#    b,g,rr=open_beam()
#    del b,g
    r=r*2*rmax/data.shape[0] #puts r in degrees 
    return r 


def radial_average (data , rmax, bins=300): #puts profile into annular bins, then interps
    
    #ang_radius=get_ang_radius()
#    b,g,rad=open_beam()
    r=get_r_matrix(data,rmax)
    radialprof, b1 =np.histogram(np.ravel(r), bins=bins, weights=np.ravel(data))
    nr, b2=np.histogram(np.ravel(r),bins=bins) #not sure if this is necessary
    index=nr==0
    nr[index]=1 #to avoid divide by zero
    radialprof = radialprof / nr
    
    f = interp1d(b1[:-1], radialprof)

    return radialprof,f

def open_beam(filename='beam_profile_160201_2014_pa2_jitmap_CMB_deep56.txt'):#ensure filename is a string
    fd = open(filename, 'rb')
    b=np.loadtxt(fd, dtype=np.float32, skiprows=6, usecols=(0,1))
    fd.close()
    r=b[:,0]
    beam=b[:,1]
    g=interp1d(r,beam)
    return beam, g, r.max()

def get_2D_profiles(profile):
    #beam
    b,g,r=open_beam()
    del b
    rr=get_r_matrix(profile,r)
    rmatrix=np.where(rr>r, np.NaN, rr)
    beam=g(rmatrix.flat).reshape(rmatrix.shape)
    beam=np.nan_to_num(beam)
    
    ang_radius=get_ang_radius()
    rn=np.where(rr>ang_radius, np.NaN, rr)
    radprof,f=radial_average(profile, ang_radius)
    del radprof
    profile=f(rn.flat).reshape(rn.shape)
    profile=np.nan_to_num(profile)
    
    return beam, profile




#prof1= open_profile('GEN_Cluster_35L165.256.FBN2_snap54_comovFINE.d')
#prof2= open_profile('GEN_Cluster_64L165.256.FBN2_snap54_comovFINE.d')
#prof3= open_profile('GEN_Cluster_98L165.256.FBN2_snap54_comovFINE.d')
#prof4= open_profile('GEN_Cluster_99L165.256.FBN2_snap54_comovFINE.d')
#prof5= open_profile('GEN_Cluster_206L165.256.FBN2_snap54_comovFINE.d')
#prof6= open_profile('GEN_Cluster_269L165.256.FBN2_snap54_comovFINE.d')

prof1= open_profile('GEN_Cluster_118L165.256.FBN2_snap47_comovFINE.d')
prof2= open_profile('GEN_Cluster_201L165.256.FBN2_snap47_comovFINE.d')
prof3= open_profile('GEN_Cluster_205L165.256.FBN2_snap47_comovFINE.d')
prof4= open_profile('GEN_Cluster_271L165.256.FBN2_snap47_comovFINE.d')
prof5= open_profile('GEN_Cluster_63L165.256.FBN2_snap47_comovFINE.d')
prof6= open_profile('GEN_Cluster_95L165.256.FBN2_snap47_comovFINE.d')




#opening 6 random cluster profiles


#Taking the element-wise average of the 6 profiles
average= np.mean(np.array([prof1,prof2,prof3,prof4,prof5,prof6]),axis=0)


##plt.imshow(average)
#
##radial_ave,f= radial_average(average)
#cmbmap=np.zeros((1600,4000)) #array with shape of cmb map

beam,profile= get_2D_profiles(average)

wbeam,wprofile=np.fft.fftshift(beam), np.fft.fftshift(profile)

wconvolve= np.real(np.fft.ifft2(np.fft.fft2(wbeam)/np.sum(wbeam)*np.fft.fft2(wprofile)))

convolve= np.fft.fftshift(wconvolve)

b,g,r=open_beam()

radprof, func =radial_average(convolve,r)



pix=60
yy,xx=np.indices((pix,pix))*30/3600 #in degrees
cent=yy[int(pix/2),int(pix/2)]

rmat = np.sqrt((xx - cent)**2 + (yy - cent)**2)
rmatd=rmat#/3600
something=func(rmatd.flat).reshape(rmatd.shape)
something =something*2.73*1000000*-0.95



act150=fits.getdata('weightedMap_4.fits')[ymin:ymax,xmin:xmax]
pl220=fits.getdata('HFI_equD56_217.fits')[ymin:ymax,xmin:xmax]*1000000

#diff=act150-pl220


ncomp=1
nfreq=2
m=ymax-ymin #or act150.shape[0]
n=xmax-xmin #or act150.shape[1]

ydiff=m-something.shape[0]
xdiff=n-something.shape[1]

paddedtemp= np.pad(something,((int(ydiff/2),int(ydiff/2)),(int(xdiff/2),int(xdiff/2))), 'constant' )

temp=np.fft.fftshift(paddedtemp)

fttemp=np.fft.fft2(temp)


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

ninv_a=np.zeros((2*m,n))
ninv_a[:m,:]=(np.divide(fttemp*d-0*b, a*d-b*b))
ninv_a[m:,:]=(np.divide(-fttemp*b+0*a, a*d-b*b))


#ordering ninv_a as we had ordered A previously .      
#ninv_a=np.reshape(ninv_a, (2*m*n,ncomp),order='C')


ninv_a=np.fft.ifft2(ninv_a).reshape((2*m*n,1),order='C')

A=np.zeros((2*m*n,1))
A[:m*n,:]=np.reshape(temp,(m*n,1))
A[m*n:,:]=0


#lhs=A.transpose()*ninv_a #- gives error:iterator is too large 

lhs=np.real(np.sum(A*ninv_a)) 


ninv_d1=np.divide(ft150*d-ft220*b,a*d-b*b)
ninv_d2=np.divide(-ft150*b+ft220*a,a*d-b*b)

del a,b,d, # ft150, ft220

rhs=np.real(np.fft.ifft2(fttemp*(ninv_d1)+0*(ninv_d2)))
rhs=np.ravel(rhs)

fit=(np.divide(rhs,lhs))

actplanck=(fit.reshape(m,n))

tempnorm=temp/np.sum(temp)

#smooth by convolving with a gaussian
actplanck=np.real(np.fft.ifft2(np.fft.fft2(actplanck)*np.fft.fft2(tempnorm)))

apsigma=np.std(actplanck)

#ii=np.argwhere(actplanck>thresh*apsigma)
#ii[:,0]=ii[:,0]+ymin
#ii[:,1]=ii[:,1]+xmin
#np.savetxt(filename, ii)

def apsnr(x,y,sigma=apsigma):
    x=x-xmin
    y=y-ymin
    apsnr=actplanck[y,x]/sigma
    return apsnr