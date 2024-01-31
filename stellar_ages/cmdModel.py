import numpy as np
import minimint
import matplotlib.pyplot as plt
from scipy import interpolate
from PyAstronomy import pyasl


rsun = 69634000000. # radius of the sun in cm
msun = 1.989*10**33 # mass of the sun in g
G = 6.67*10**-8 # cgs units gravitational constant

# define filters used for CMD -- assumes GaiaDR3 values used -- need to make this an optional feature 
filters = ['Gaia_G_EDR3', 'Gaia_BP_EDR3', 'Gaia_RP_EDR3']
ii = minimint.Interpolator(filters)

def get_dereddened_mags(self,theta):
    ebmv = theta[np.where(self.labels=='ebmv')[0]][0] # get ebmv value from theta array
    wvls = np.array([self.gaia_wvl, self.gaia_bp_wvl, self.gaia_rp_wvl]) 
    mags = np.array([self.gaia_mag, self.gaia_bp, self.gaia_rp])
    # calculate dereddened magnitudes 
    dereddened_mags = -2.5*np.log10(pyasl.unred(wvls, 10**(mags/-2.5), ebv=ebmv, R_V=3.1))
    return dereddened_mags

def getcmd(self,theta, bmr_obs, gmag_obs):

    age = theta[np.where(self.labels=='age')[0]][0] # get age from theta array
    mstar = np.arange(0.3,5.0,0.01) # simulate a range of stars from 0.3 to 5 solar masses at 0.01 solar mass intervals
    feh = np.zeros(len(mstar))
    age = age * np.ones(len(mstar))
    iso = ii(mstar, np.log10(age*10**6), feh) # compute MIST model 
    
    bpmrp = iso["Gaia_BP_EDR3"] - iso['Gaia_RP_EDR3'] # calculate modeled Gaia BP-RP
    gmag = iso['Gaia_G_EDR3'] # get modeled Gaia Mag

    # filter for nans
    mask1 = bpmrp == bpmrp 
    mask2 = gmag == gmag
    mask = mask1 & mask2
    bpmrp = bpmrp[mask]
    gmag = gmag[mask]

    
    indx = np.argsort(bpmrp)
    bpmrp,gmag = bpmrp[indx],gmag[indx]
    interp = interpolate.splrep(bpmrp,gmag,k=1)

    g_model = interpolate.splev(bmr_obs,interp)

    return g_model
