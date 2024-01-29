import pandas as pd
from fit_stellar_ages import *

def load_priors(file_path):
    tab = pd.read_csv(file_path)
    tab=tab
    labels = tab.parameters.values#.astype(str)
    theta0 = tab.initial_guess.values.astype(float)
    prior_type = tab.prior.values#.astype(str)
    bounds_1 = tab.prior1.values.astype(float)
    bounds_2 = tab.prior2.values.astype(float)
    priors = np.vstack((prior_type,bounds_1, bounds_2))
#    print(priors)
    return labels, theta0, priors

tab = pd.read_csv('~/Research/unisq/TOI5358/gyro_stars.csv')
rotations = tab[tab.prot > 0.5] 
rotations = rotations[rotations.prot < 12] 
rotations = rotations[rotations.teff < 6500]
# rotations = rotations[rotations.teff > 3800]
rotations = rotations[rotations.rv_err < 5]
rotations = rotations[rotations.ruwe < 1.4]

rotations.teff = rotations.teff.values.astype(float)
rotations.prot = rotations.prot.values.astype(float)

labels,theta,priors=load_priors('priors.csv')

li = pd.read_csv('/Users/u1147646/Downloads/li_tres_ew.csv')
pl_li = li[li.assoc2 == 'pleiades']
th_li = li[~li.tic.isin(pl_li.tic.values)]
th_li = th_li[th_li.assoc == 'theia369']

th_li = th_li.sort_values('teff')
th_li = th_li[th_li.teff < 6500]
th_li = th_li[th_li.teff > 3000]
print(th_li.teff, -th_li.ew * 1000, th_li.ew_err * 1000)
print(th_li.columns)

# csv2 = th_li.loc[:, ['tic', 'teff', 'ew', 'ew_err']]
# csv2.ew = csv2.ew.values * -1000
# csv2.ew_err = csv2.ew_err * 1000
# csv2.to_csv('input.dat', index=False)
# b = c

cluster = fitStellarCluster(labels, priors, \
                            measured_rotations = rotations.prot.values, \
                            measured_rotations_err = np.ones(len(rotations)) * 0.5, \
                            rot_teff = rotations.teff.values, rot_model = "bouma+2023",\
                            li=th_li.ew.values, li_err=th_li.ew_err.values, \
                            teff=th_li.teff.values)

cluster.fit_ages(20, theta)

