import minimint
import numpy as np

rsun = 69634000000.
msun = 1.989*10**33
G = 6.67*10**-8

filters = ['Tycho_B','Tycho_V', "2MASS_J","2MASS_H","2MASS_Ks","Gaia_G_DR2Rev","Gaia_BP_EDR3", 'Gaia_RP_EDR3']
ii = minimint.Interpolator(filters)



def calculate_sed_model(mstar, age, feh, parallax, obs_mags, obs_mags_e):
    iso = ii(mstar, np.log10(age*10**6), feh)
    radius = np.sqrt((G*mstar*msun) / (10**iso['logg'][0])) / rsun

    teff_iso = 10**iso['logteff']

    mags = [
        iso['Tycho_B'][0],
        iso['Tycho_V'][0],
        iso['2MASS_J'][0],
        iso['2MASS_H'][0],
        iso['2MASS_Ks'][0],
        iso['Gaia_G_DR2Rev'][0],
        iso['Gaia_BP_EDR3'][0],
        iso['Gaia_RP_EDR3'][0]
    ]
    mags = np.array(mags)

    dist = 1/(0.001*parallax)
    dist_mod = 5*np.log10(dist)-5

    return dist_mod