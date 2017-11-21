#!/usr/bin/env cdat
"""
% Cloud Property Histograms. Part I: Cloud Radiative Kernels. J. Climate, 25, 3715?3735. doi:10.1175/JCLI-D-11-00248.1.

% v2: This script is written to demonstrate how to compute the cloud feedback using for a 
% short (2-year) period of MPI-ESM-LR using the difference between amipFuture and amip runs.
% One should difference longer periods for more robust results -- these are just for demonstrative purposes

% Data that are used in this script:
% 1. model clisccp field
% 2. model rsuscs field
% 3. model rsdscs field
% 4. model tas field
% 5. cloud radiative kernels

% This script written by Mark Zelinka (zelinka1@llnl.gov) on 14 July 2017

"""

#IMPORT STUFF:
#=====================
import cdms2 as cdms2
import glob
import cdutil
import MV2 as MV
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#import matplotlib.cmoloas cm

###########################################################################
# HELPFUL FUNCTIONS FOLLOW 
###########################################################################

###########################################################################
def add_cyclic(data):
    # Add Cyclic point around 360 degrees longitude:
    lons=data.getLongitude()[:]
    dx=np.gradient(lons)[-1]
    data2 = data(longitude=(0, dx+np.max(lons)), squeeze=True)    
    return data2

###########################################################################
def nanarray(vector):

    # this generates a masked array with the size given by vector
    # example: vector = (90,144,28)

    # similar to this=NaN*ones(x,y,z) in matlab

    this=MV.zeros(vector)
    this=MV.masked_where(this==0,this)

    return this

###########################################################################
def map_SWkern_to_lon(Ksw,albcsmap):

    from scipy.interpolate import interp1d
    ## Map each location's clear-sky surface albedo to the correct albedo bin
    # Ksw is size 12,7,7,lats,3
    # albcsmap is size A,lats,lons
    albcs=np.arange(0.0,1.5,0.5) 
    A=albcsmap.shape[0]
    TT=Ksw.shape[1]
    PP=Ksw.shape[2]
    lenlat=Ksw.shape[3]
    lenlon=albcsmap.shape[2]
    SWkernel_map=nanarray((A,TT,PP,lenlat,lenlon))
    for M in range(A):
        MM=M
        while MM>11:
            MM=MM-12
        for LA in range(lenlat):
            alon=albcsmap[M,LA,:] 
            # interp1d can't handle mask but it can deal with NaN (?)
            try:
                alon2=MV.where(alon.mask,np.nan,alon)   
            except:
                alon2=alon
            if np.ma.count(alon2)>1: # at least 1 unmasked value
                if len(pl.find(Ksw[MM,:,:,LA,:]>0))==0:
                    SWkernel_map[M,:,:,LA,:] = 0
                else:
                    f = interp1d(albcs,Ksw[MM,:,:,LA,:],axis=2)
                    ynew = f(alon2.data)
                    ynew=MV.masked_where(alon2.mask,ynew)
                    SWkernel_map[M,:,:,LA,:] = ynew
            else:
                continue

    return SWkernel_map

###########################################################################
# MAIN ROUTINE FOLLOWS
###########################################################################
direc='/Users/cpatrizio/repos/cloud-radiative-kernels/data/'
fin = '/Users/cpatrizio/data/CMIP5/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/CMIP5figs/'

model_names =['MRI-CGCM3']

print 'model:', model_names[0]

if model_names[0] == 'MIROC5':
    year1='21'
    year2='22'
else:
    year1='18'
    year2='19'

# Load in the Zelinka et al 2012 kernels:
f=cdms2.open(direc+'cloud_kernels2.nc')
LWkernel=f('LWkernel')
SWkernel=f('SWkernel')
f.close()

albcs=np.arange(0.0,1.5,0.5) # the clear-sky albedos over which the kernel is computed

# LW kernel does not depend on albcs, just repeat the final dimension over longitudes:
LWkernel_map=np.tile(np.tile(LWkernel[:,:,:,:,0],(1,1,1,1,1)),(144,1,1,1,1))(order=[1,2,3,4,0])

# Define the cloud kernel axis attributes
lats=LWkernel.getLatitude()[:]
nlats = len(lats)
lons=np.arange(1.25,360,2.5)
nlons = len(lons)
grid = cdms2.createGenericGrid(lats,lons)

# Load in clisccp from two models
#f=cdms2.open(direc+'clisccp_cfMon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc','r')
f=cdms2.open(glob.glob(fin + model_names[0] + '/clisccp*_' + year1 + '*.nc')[0],'r')
clisccp1=f('clisccp')
t1i = clisccp1.getTime()[0]
t2i = clisccp1.getTime()[-1]
print 'clisccp1.shape', clisccp1.shape

f.close()
#f=cdms2.open(direc+'clisccp_cfMon_MPI-ESM-LR_amipFuture_r1i1p1_197901-198112.nc','r')
f=cdms2.open(glob.glob(fin + model_names[0] + '/clisccp*_' + year2 + '*.nc')[0],'r')
clisccp2=f('clisccp')
t1f = clisccp2.getTime()[0]
t2f = clisccp2.getTime()[-1]
print 'clisccp2.shape', clisccp2.shape
f.close()

# Make sure clisccp is in percent  
sumclisccp1=MV.sum(MV.sum(clisccp1,2),1)
sumclisccp2=MV.sum(MV.sum(clisccp2,2),1)   
if np.max(sumclisccp1) <= 1.:
    clisccp1 = clisccp1*100.        
if np.max(sumclisccp2) <= 1.:
    clisccp2 = clisccp2*100.

# Compute climatological annual cycle:
avgclisccp1=cdutil.ANNUALCYCLE.climatology(clisccp1) #(12, TAU, CTP, LAT, LON)
avgclisccp2=cdutil.ANNUALCYCLE.climatology(clisccp2) #(12, TAU, CTP, LAT, LON)
del(clisccp1,clisccp2)

# Compute clisccp anomalies
anomclisccp = avgclisccp2 - avgclisccp1

# Compute clear-sky surface albedo
#f=cdms2.open(direc+'rsuscs_Amon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc','r')
f=cdms2.open(glob.glob(fin + model_names[0] + '/rsuscs*.nc')[0],'r')
rsuscs1 = f('rsuscs', time=(t1i, t2i))
f.close()
#f=cdms2.open(direc+'rsdscs_Amon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc','r')
f=cdms2.open(glob.glob(fin+ model_names[0] + '/rsdscs*.nc')[0],'r')
rsdscs1 = f('rsdscs', time=(t1i, t2i))
f.close()

albcs1=rsuscs1/rsdscs1
avgalbcs1=cdutil.ANNUALCYCLE.climatology(albcs1) #(12, 90, 144)
avgalbcs1=MV.where(avgalbcs1>1.,1,avgalbcs1) # where(condition, x, y) is x where condition is true, y otherwise
avgalbcs1=MV.where(avgalbcs1<0.,0,avgalbcs1)
del(rsuscs1,rsdscs1,albcs1)

# Load surface air temperature
#f=cdms2.open(direc+'tas_Amon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc','r')
f=cdms2.open(glob.glob(fin + model_names[0] + '/tas*.nc')[0],'r')
tas1 = f('tas', time=(t1i,t2i))
print 'tas1.shape', tas1.shape
#f.close()

#f=cdms2.open(direc+'tas_Amon_MPI-ESM-LR_amipFuture_r1i1p1_197901-198112.nc','r')
tas2 = f('tas', time=(t1f, t2f))
print 'tas2.shape', tas2.shape
f.close()

# Compute climatological annual cycle:
avgtas1=cdutil.ANNUALCYCLE.climatology(tas1) #(12, 90, 144)
avgtas2=cdutil.ANNUALCYCLE.climatology(tas2) #(12, 90, 144)
del(tas1,tas2)

# Compute global annual mean tas anomalies
anomtas = avgtas2 - avgtas1
avgdtas = cdutil.averager(MV.average(anomtas,axis=0), axis='xy', weights='weighted') # (scalar)

# Regrid everything to the kernel grid:
avgalbcs1 = add_cyclic(avgalbcs1)
avgclisccp1 = add_cyclic(avgclisccp1)
avgclisccp2 = add_cyclic(avgclisccp2)
avganomclisccp = add_cyclic(anomclisccp)
avgalbcs1_grd = avgalbcs1.regrid(grid,regridTool="esmf",regridMethod = "linear")
avgclisccp1_grd = avgclisccp1.regrid(grid,regridTool="esmf",regridMethod = "linear")
avgclisccp2_grd = avgclisccp2.regrid(grid,regridTool="esmf",regridMethod = "linear")
avganomclisccp_grd = avganomclisccp.regrid(grid,regridTool="esmf",regridMethod = "linear")

#Decomposition of cloud amount anomaly: initialize matrices for DeltaC_prop, DeltaC_tau and DeltaC_p
deltaC_prop = nanarray(avgclisccp1_grd.shape)
#deltaC_prop[:] = np.nan
#deltaC_prop = MV.masked_equal(deltaC_prop,0)
avedeltaC_p = nanarray(avgclisccp1_grd.shape)
#avedeltaC_p[:] = np.nan
#avedeltaC_p = MV.masked_equal(avedeltaC_p,0)
avedeltaC_tau = nanarray(avgclisccp1_grd.shape)
#avedeltaC_tau[:] = np.nan
#avedeltaC_tau = MV.masked_equal(avedeltaC_tau,0)

#Calculate total cloud amount and total cloud amount anomaly matrices (12, 90, 144)
C_tot = MV.sum(MV.sum(avgclisccp1_grd, axis=2), axis=1)
deltaC_tot = MV.sum(MV.sum(avganomclisccp_grd, axis=2), axis=1)

#Calculate average matrices to subtract from cloud amount anomaly matrix (12, 7, 7, 90, 144)
for i in np.arange(12):
    deltaC_prop[i,:] = avgclisccp1_grd[i,:]*(deltaC_tot[i,:]/C_tot[i,:])
    avedeltaC_p[i,:] = np.mean(avganomclisccp_grd[i,:], axis=1, keepdims=True)
    avedeltaC_tau[i,:] = np.mean(avganomclisccp_grd[i,:], axis=0, keepdims=True)
    
P = avgclisccp1_grd.shape[2]
T = avgclisccp1_grd.shape[1]

#Calculate DeltaC_prop, DeltaC_tau and DeltaC_p matrices 
deltaC_p = avganomclisccp_grd - avedeltaC_p
deltaC_tau = avganomclisccp_grd - avedeltaC_tau
deltaC_totdecomp = deltaC_p + deltaC_tau + deltaC_prop
deltaC_res = avganomclisccp_grd - deltaC_totdecomp

#the sum of deltaC_totdecomp for a given month/location should be equal to that of avganomclisccp_grd
#the sum of residual for a given month/location should be zero
print 'sum deltaC_prop:', MV.sum(deltaC_prop[1,:,:,1,1])
print 'sum deltaC_totdecomp:', MV.sum(deltaC_totdecomp[1,:,:,1,1])
print 'sum avganomclisccp_grd:', MV.sum(avganomclisccp_grd[1,:,:,1,1])
print 'sum deltaC_res:', MV.sum(deltaC_res[1,:,:,1,1])

#MV.average(avgclisccp1_grd)

# Use control albcs to map SW kernel to appropriate longitudes
SWkernel_map = map_SWkern_to_lon(SWkernel,avgalbcs1_grd)

# Compute clisccp anomalies normalized by global mean delta tas
anomclisccp = avganomclisccp_grd/avgdtas

# Compute feedbacks: Multiply clisccp anomalies by kernels
SW0 = SWkernel_map*anomclisccp
LW_cld_fbk = LWkernel_map*anomclisccp
LW_cld_fbk.setAxisList(anomclisccp.getAxisList())

LW_cld_fbk_deltaC_prop = LWkernel_map*(deltaC_prop/avgdtas)
LW_cld_fbk_deltaC_p = LWkernel_map*(deltaC_p/avgdtas)
LW_cld_fbk_deltaC_tau = LWkernel_map*(deltaC_tau/avgdtas)
LW_cld_fbk_deltaC_res = LWkernel_map*(deltaC_res/avgdtas)

SW_cld_fbk_deltaC_prop = SWkernel_map*(deltaC_prop/avgdtas)
SW_cld_fbk_deltaC_p = SWkernel_map*(deltaC_p/avgdtas)
SW_cld_fbk_deltaC_tau = SWkernel_map*(deltaC_tau/avgdtas)
SW_cld_fbk_deltaC_res= SWkernel_map*(deltaC_res/avgdtas)

# Set the SW cloud feedbacks to zero in the polar night
# The sun is down if every bin of the SW kernel is zero:
sundown=MV.sum(MV.sum(SWkernel_map,axis=2),axis=1)  #12,90,144
repsundown=np.tile(np.tile(sundown,(1,1,1,1,1)),(7,7,1,1,1))(order=[2,1,0,3,4])
SW1 = MV.where(repsundown==0, 0, SW0) # where(condition, x, y) is x where condition is true, y otherwise
SW_cld_fbk = MV.where(repsundown.mask, 0, SW1) # where(condition, x, y) is x where condition is true, y otherwise
SW_cld_fbk.setAxisList(anomclisccp.getAxisList())

# SW_cld_fbk and LW_cld_fbk contain the contributions to the feedback from cloud anomalies in each bin of the histogram

# Quick sanity check:
# print the global, annual mean LW and SW cloud feedbacks:
sumLW = MV.average(MV.sum(MV.sum(LW_cld_fbk,axis=2),axis=1),axis=0)
avgLW_cld_fbk = cdutil.averager(sumLW, axis='xy', weights='weighted')
print 'avg LW cloud feedback = '+str(avgLW_cld_fbk)
sumSW = MV.average(MV.sum(MV.sum(SW_cld_fbk,axis=2),axis=1),axis=0)
avgSW_cld_fbk = cdutil.averager(sumSW, axis='xy', weights='weighted')
print 'avg SW cloud feedback = '+str(avgSW_cld_fbk)

tau=[0.,0.3,1.3,3.6,9.4,23.,60.,380.]
ctp=[1000,800,680,560,440,310,180,50]

## amip cloud fraction histogram:
#plt.subplots()
#data = cdutil.averager(MV.average(avgclisccp1_grd,axis=0), axis='xy', weights='weighted').transpose()
#plt.pcolormesh(data,shading='flat',cmap='Blues_r',vmin=0, vmax=10)
#plt.xticks(np.arange(8), tau)
#plt.yticks(np.arange(8), ctp)
#plt.title('Global mean amip cloud fraction')
#plt.xlabel(r'$\tau$')
#plt.ylabel('CTP')
#plt.colorbar()
#
## amipFuture cloud fraction histogram:
#plt.subplots()
#data = cdutil.averager(MV.average(avgclisccp2_grd,axis=0), axis='xy', weights='weighted').transpose()
#plt.pcolormesh(data,shading='flat',cmap='Blues_r',vmin=0, vmax=10)
#plt.xticks(np.arange(8), tau)
#plt.yticks(np.arange(8), ctp)
#plt.title('Global mean amipFuture cloud fraction')
#plt.xlabel(r'$\tau$')
#plt.ylabel('CTP')
#plt.colorbar()

# difference of cloud fraction histograms:
fig = plt.figure(figsize=(7,11))
#plt.subplots()
ax = fig.add_subplot(311)
data = cdutil.averager(MV.average(anomclisccp,axis=0), axis='xy', weights='weighted').transpose()
plt.pcolormesh(data,shading='flat',cmap='RdBu_r',vmin=-0.75, vmax=0.75)
plt.xticks(np.arange(8), tau)
plt.yticks(np.arange(8), ctp)
plt.title('Global mean change in cloud fraction')
#plt.xlabel(r'$\tau$')
plt.ylabel('CTP')
plt.colorbar(label=r'% K$^{{-1}}$')


# LW cloud feedback contributions:
ax = fig.add_subplot(312)
data = cdutil.averager(MV.average(LW_cld_fbk,axis=0), axis='xy', weights='weighted').transpose()
plt.pcolormesh(data,shading='flat',cmap='RdBu_r',vmin=-0.75, vmax=0.75)
plt.xticks(np.arange(8), tau)
plt.yticks(np.arange(8), ctp)
plt.title('Global mean LW cloud feedback')
#plt.xlabel(r'$\tau$')
plt.ylabel('CTP')
plt.colorbar(label=r'W m$^{{-2}}$ K$^{{-1}}$')


# SW cloud feedback contributions:
ax = fig.add_subplot(313)
data = cdutil.averager(MV.average(SW_cld_fbk,axis=0), axis='xy', weights='weighted').transpose()
plt.pcolormesh(data,shading='flat',cmap='RdBu_r',vmin=-0.75, vmax=0.75)
plt.xticks(np.arange(8), tau)
plt.yticks(np.arange(8), ctp)
plt.title('Global mean SW cloud feedback')
plt.xlabel(r'$\tau$')
plt.ylabel('CTP')
plt.colorbar(label=r'W m$^{{-2}}$ K$^{{-1}}$')
#plt.savefig(fout + 'cldfbks_hist_MPI_ESM_amip.pdf')
plt.suptitle(model_names[0])
plt.savefig(fout + 'cldfbks_hist_' + model_names[0] + '_abrupt4xCO2.pdf')
plt.close()

#Plot maps of cloud anomalies and feedbacks
anomclisccp_grid = MV.average(MV.sum(MV.sum(anomclisccp, axis=2), axis=1), axis=0)
LW_cld_fbk_grid = MV.average(MV.sum(MV.sum(LW_cld_fbk, axis=2), axis=1), axis=0)
SW_cld_fbk_grid = MV.average(MV.sum(MV.sum(SW_cld_fbk, axis=2), axis=1), axis=0)

#decomposition of deltaC into a contribution from cloud amount, CTP and optical depth
LW_cld_fbk_deltaC_prop_grid = MV.average(MV.sum(MV.sum(LW_cld_fbk_deltaC_prop, axis=2), axis=1), axis=0)
LW_cld_fbk_deltaC_p_grid = MV.average(MV.sum(MV.sum(LW_cld_fbk_deltaC_p, axis=2), axis=1), axis=0)
LW_cld_fbk_deltaC_tau_grid = MV.average(MV.sum(MV.sum(LW_cld_fbk_deltaC_tau, axis=2), axis=1), axis=0)
LW_cld_fbk_deltaC_res_grid = MV.average(MV.sum(MV.sum(LW_cld_fbk_deltaC_res, axis=2), axis=1), axis=0)


#decomposition of deltaC into a contribution from cloud amount, CTP and optical depth
SW_cld_fbk_deltaC_prop_grid = MV.average(MV.sum(MV.sum(SW_cld_fbk_deltaC_prop, axis=2), axis=1), axis=0)
SW_cld_fbk_deltaC_p_grid = MV.average(MV.sum(MV.sum(SW_cld_fbk_deltaC_p, axis=2), axis=1), axis=0)
SW_cld_fbk_deltaC_tau_grid = MV.average(MV.sum(MV.sum(SW_cld_fbk_deltaC_tau, axis=2), axis=1), axis=0)
SW_cld_fbk_deltaC_res_grid = MV.average(MV.sum(MV.sum(SW_cld_fbk_deltaC_res, axis=2), axis=1), axis=0)


cldanom_max = 8
LW_max = 15
SW_max = 15
LWdecomp_max = 10

cldlevels = np.linspace(-cldanom_max, cldanom_max, 4*cldanom_max+1)
LWlevels = np.linspace(-LW_max, LW_max, 2*LW_max+1)
SWlevels = np.linspace(-SW_max, SW_max, 2*SW_max+1)
LWdecomplevels = np.linspace(-LWdecomp_max, LWdecomp_max, 2*LWdecomp_max+1)

par = np.arange(-90.,91.,30.)
mer = np.arange(-180.,181.,60.)

plt.figure(figsize=(6,4))
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[1,0], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[1,0], linewidth=0.1)
x, y = m(*np.meshgrid(lons, lats))
levels=np.linspace(-16, 16, 65)
m.contourf(x, y, anomclisccp_grid, cmap=plt.cm.RdBu_r, levels=cldlevels)
m.colorbar(label=r'% K$^{{-1}}$', format='%3.1f')
#m.drawcountries()
#m.bluemarble()
#m.fillcontinents(color='white')
#m.drawmapboundary()
plt.title(r'$\Delta C_{{tot}}$ normalized by global mean $\Delta T_s$ = {:2.1f} K'.format(float(avgdtas.getValue())))
#plt.savefig(fout + 'cldanom_map_MPI_ESM_amip.pdf')
plt.suptitle(model_names[0])
plt.savefig(fout + 'cldanom_map_' + model_names[0] + '_abrupt4xCO2.pdf')
plt.close()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(211)
#plt.subplots()
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[1,0], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[1,0], linewidth=0.1)
x, y = m(*np.meshgrid(lons, lats))
m.contourf(x, y, LW_cld_fbk_grid, cmap=plt.cm.RdBu_r, levels=LWlevels)
m.colorbar(label=r'W m$^{{-2}}$ K$^{{-1}}$', format='%3.1f')
ax.set_title('LW cloud feedback')
#m.drawcountries()
#m.bluemarble()
#m.fillcontinents(color='white')
#m.drawmapboundary()
#plt.savefig(fout + 'test_basemap.pdf')
#plt.close()

ax = fig.add_subplot(212)
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[1,0], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[1,0], linewidth=0.1)
x, y = m(*np.meshgrid(lons, lats))
m.contourf(x, y, SW_cld_fbk_grid, cmap=plt.cm.RdBu_r, levels=SWlevels)
m.colorbar(label=r'W m$^{{-2}}$ K$^{{-1}}$', format='%3.1f')
ax.set_title('SW cloud feedback')
#plt.savefig(fout + 'cldfbks_map_MPI_ESM_amip.pdf')
plt.suptitle(model_names[0])
plt.savefig(fout + 'cldfbks_map_' + model_names[0] + '_abrupt4xCO2.pdf')
plt.close()

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(221)
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[1,0], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[1,0], linewidth=0.1)
x, y = m(*np.meshgrid(lons, lats))
m.contourf(x, y, LW_cld_fbk_deltaC_prop_grid, cmap=plt.cm.RdBu_r, levels=LWdecomplevels)
m.colorbar(label=r'W m$^{{-2}}$ K$^{{-1}}$', format='%3.1f')
ax.set_title(r'Amount')

ax = fig.add_subplot(222)
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[1,0], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[1,0], linewidth=0.1)
x, y = m(*np.meshgrid(lons, lats))
m.contourf(x, y, LW_cld_fbk_deltaC_p_grid, cmap=plt.cm.RdBu_r, levels=LWdecomplevels)
#m.colorbar(label=r'W m$^{{-2}}$ K$^{{-1}}$', format='%3.1f')
ax.set_title(r'Altitude')

ax = fig.add_subplot(223)
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[1,0], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[1,0], linewidth=0.1)
x, y = m(*np.meshgrid(lons, lats))
m.contourf(x, y, LW_cld_fbk_deltaC_tau_grid, cmap=plt.cm.RdBu_r, levels=LWdecomplevels)
m.colorbar(label=r'W m$^{{-2}}$ K$^{{-1}}$', format='%3.1f')
ax.set_title(r'Optical Depth')

ax = fig.add_subplot(224)
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[1,0], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[1,0], linewidth=0.1)
x, y = m(*np.meshgrid(lons, lats))
m.contourf(x, y, LW_cld_fbk_deltaC_res_grid, cmap=plt.cm.RdBu_r, levels=LWdecomplevels)
#m.colorbar(label=r'W m$^{{-2}}$ K$^{{-1}}$', format='%3.1f')
ax.set_title(r'Residual')
#plt.savefig(fout + 'cldfbks_LWcomponents_map_MPI_ESM_amip.pdf')
plt.suptitle(model_names[0] + ', LW Feedback')
plt.savefig(fout + 'cldfbks_LWcomponents_map_' + model_names[0] + '_abrupt4xCO2.pdf')

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(221)
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[1,0], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[1,0], linewidth=0.1)
x, y = m(*np.meshgrid(lons, lats))
m.contourf(x, y, SW_cld_fbk_deltaC_prop_grid, cmap=plt.cm.RdBu_r, levels=LWdecomplevels)
m.colorbar(label=r'W m$^{{-2}}$ K$^{{-1}}$', format='%3.1f')
ax.set_title(r'Amount')

ax = fig.add_subplot(222)
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[1,0], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[1,0], linewidth=0.1)
x, y = m(*np.meshgrid(lons, lats))
m.contourf(x, y, SW_cld_fbk_deltaC_p_grid, cmap=plt.cm.RdBu_r, levels=LWdecomplevels)
#m.colorbar(label=r'W m$^{{-2}}$ K$^{{-1}}$', format='%3.1f')
ax.set_title(r'Altitude')

ax = fig.add_subplot(223)
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[1,0], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[1,0], linewidth=0.1)
x, y = m(*np.meshgrid(lons, lats))
m.contourf(x, y, SW_cld_fbk_deltaC_tau_grid, cmap=plt.cm.RdBu_r, levels=LWdecomplevels)
m.colorbar(label=r'W m$^{{-2}}$ K$^{{-1}}$', format='%3.1f')
ax.set_title(r'Optical Depth')

ax = fig.add_subplot(224)
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[1,0], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[1,0], linewidth=0.1)
x, y = m(*np.meshgrid(lons, lats))
m.contourf(x, y, SW_cld_fbk_deltaC_res_grid, cmap=plt.cm.RdBu_r, levels=LWdecomplevels)
#m.colorbar(label=r'W m$^{{-2}}$ K$^{{-1}}$', format='%3.1f')
ax.set_title(r'Residual')
#plt.savefig(fout + 'cldfbks_SWcomponents_map_MPI_ESM_amip.pdf')
plt.suptitle(model_names[0] + ', SW Feedback')
plt.savefig(fout + 'cldfbks_SWcomponents_map_' + model_names[0] + '_abrupt4xCO2.pdf')



##Plot map of cloud anomalies using VCS
#avganomclisccp = MV.average(MV.sum(MV.sum(anomclisccp, axis=2), axis=1), axis=0)
#iso = vcs.createisofill()
##box.projection = "robinson"
#x = vcs.init()
##rob = x.getisofill('robinson')
#x.setcontinentstype(4)
#x.setcontinentsline('thick')
#pm = x.createprojection()
#pm.type = 21
#x.plot(avganomclisccp, pm, iso, units=r'% K$^{{-1}}$', name='mean cloud amount anomaly (amipFuture - amip)', file_comment='MPI-ESM-LR_amip, 1979-01 to 1981-12', xname = 'longitude', yname='latitude')
##x.plot(grid, box)
#x.pdf(fout + 'MPI_ESM_amip_cldanom_map.pdf')
#x.close()






