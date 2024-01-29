import minimint
import emcee
from scipy.optimize import minimize
import pickle
import numpy as np
from PyAstronomy import pyasl
from stellarRotationModel import *
from LiModel import *


rsun = 69634000000.
msun = 1.989*10**33
G = 6.67*10**-8

# filters = ['Tycho_B','Tycho_V', "2MASS_J","2MASS_H","2MASS_Ks","Gaia_G_DR2Rev","Gaia_BP_EDR3", 'Gaia_RP_EDR3']
# ii = minimint.Interpolator(filters)

class fitStellarCluster:
    def __init__(self, labels, priors, measured_rotations = None, measured_rotations_err = None, rot_model='default', rot_teff=None, bmv=None, cluster_name = None, li = None, li_err = None, teff=None):
        
        self.labels = labels
        self.rotp = measured_rotations
        self.rotp_err = measured_rotations_err
        self.li = li * -1000.
        self.li_err = li_err * 1000.
        self.priors = priors
        if cluster_name:
            self.cluster_name = cluster_name
        else:
            self.cluster_name = 'temp'
        
        
        if measured_rotations is None:
            print("No rotations being used in fit to derive cluster age.")
        if measured_rotations is not None:
            if len(measured_rotations) > 1:
                print("User provided rotation periods are being used to derive cluster age.")     
            if len(measured_rotations) == 1:
                print("User provided rotation periods are being used to derive star age.")
            
            self.rot_model = rot_model
            if rot_model == 'default':
                print("Default rotation model is being used. Default model is based on Mamajek and Hillenbrand (2008).")
                self.bmv = bmv
            if rot_model == 'bouma+2023':
                print("Bouma+2023 rotation model is being used.")
                self.rot_teff = rot_teff
            
        if li is not None:
            print("User provided lithium is being used to derive cluster age.")
            self.teff = teff
            
    def rotation_model(self,theta):
        if self.rot_model == 'default':
            return rotationModel_Mamajek2008(self,theta)
        if self.rot_model == 'bouma+2023':
            return rotationModel_Bouma2023(self,theta, plot=False)
        
    def lithium_model(self,theta):
        return eaglesLiModel(self,theta)
            
    def ln_posterior(self, theta):
        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_like(theta)
    
    def ln_prior(self, theta):
        lp=0.
        for i in range(len(theta)):
            t = theta[i]
            p = [self.priors[0][i], self.priors[1][i], self.priors[2][i]]
            if p[0] == 'uniform':
                if p[1] <= t <= p[2]:
                    lp-=0.
                else:
                    lp=-np.inf
                    return -np.inf
            if p[0] == 'gaussian':
                mu = p[1]
                sig = p[2]
                lp -= 0.5*((t-mu)/sig)**2
                if not np.isfinite(lp):
                    return -np.inf
        return lp
    
    def calc_chisq_jitter(self, obs, model, obs_jitter, error):
        num = obs - model
        denom = np.sqrt(obs_jitter**2 + error**2)
        chisq = np.nansum((num/denom)**2)
        return chisq

    def calc_lnlike_jitter(self, obs, chisq, obs_jitter, error):
        N = float(len(obs))
        first_term = N * np.log(2 * np.pi)
        second_term = chisq + np.nansum(np.log(obs_jitter**2 + error**2))
        lnlike = -0.5 * (first_term + second_term)
        return lnlike
    
    def ln_like(self, theta):
        lnlike = 0
        if self.rotp is not None:
            model = self.rotation_model(theta)
            jitter_rotp = theta[np.where(self.labels=='jitter_rotp')[0]][0]
            chisq_rotp = self.calc_chisq_jitter(self.rotp, model, jitter_rotp, self.rotp_err)
            lnlike_rotp = self.calc_lnlike_jitter(self.rotp, chisq_rotp, jitter_rotp, self.rotp_err)
            lnlike += lnlike_rotp
        if self.li is not None:
            model = self.lithium_model(theta)
            jitter_li = theta[np.where(self.labels=='jitter_li')[0]][0]
            chisq_li = self.calc_chisq_jitter(self.li, model, jitter_li, self.li_err)
            lnlike_li = self.calc_lnlike_jitter(self.li, chisq_li, jitter_li, self.li_err)

            lnlike += lnlike_li

        if lnlike != lnlike:
            lnlike = -np.inf
            
        return lnlike
    
            
    def fit_ages(self, nwalkers, theta0, nsteps=1000, burnin=500):
        ndim = len(theta0)
        theta0 = np.tile(theta0, (nwalkers, 1)) + 1e-4 * np.random.rand(nwalkers, ndim)
        self.ndim = ndim
        
        filename = self.cluster_name + '.h5'
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)
        
        print('about to run sampler')
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.ln_posterior, backend=backend)
        
#         max_n = nsteps
#         index = 0
#         autocorr = np.empty(max_n)
#         old_tau = np.inf

#         # run mcmc
#         print('running with autocorrelation time')
#         for sample in sampler.sample(theta0, iterations=max_n, progress=True):
#             if sampler.iteration % 100:
#                 continue
#             tau = sampler.get_autocorr_time(tol=0)
#             autocorr[index] = np.mean(tau)
#             index+=1

#             converged = np.all(tau*100 < sampler.iteration)
#             converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
#             if converged:
#                 break
#             old_tau = tau
            
        # get chains and flatten them
        #samples = sampler.get_chain(discard=burn_in)
#         tau = sampler.get_autocorr_time()
#         burnin = int(2 * np.max(tau))
#         thin = int(0.5 * np.min(tau))

#         samples = sampler.get_chain(discard=burnin, thin=thin)
        sampler.run_mcmc(theta0, nsteps, progress=True)
        samples = sampler.get_chain(discard=burnin)

        
        plt.figure()
        for i in range(self.ndim):
            plt.subplot(self.ndim, 1, i + 1)
            plt.plot(samples[:,:, i], alpha=0.3)
            plt.ylabel("Param {}".format(i))
        plt.xlabel("Step")
        plt.show()

        samples = sampler.get_chain(discard=burnin, flat=True)
        # Calculate median and percentiles
        derived_params = np.median(samples, axis=0)
        uncertainties = np.percentile(samples, [16, 84], axis=0) - derived_params
        
        for i in range(len(self.labels)):
            print(self.labels[i], derived_params[i], uncertainties[0][i], '/', uncertainties[1][i])
        return derived_params, uncertainties



        
            