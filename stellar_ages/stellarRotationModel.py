import numpy as np
import gyrointerp
import matplotlib.pyplot as plt

def rotationModel_Bouma2023(self,theta, plot=False):
    age = theta[np.where(self.labels=='age')[0]][0]
    model = gyrointerp.models.slow_sequence(self.rot_teff, age, verbose=False)
    if plot==True:
        plt.plot(self.rot_teff, model, "r.", label='Bouma+2023 Model')        
        plt.plot(self.rot_teff, self.rotp, "k.", label='Measured Rotations')
        plt.legend()
        plt.show()
        
    return model

def rotationModel_Mamajek2008(self,theta):
    age = theta[np.where(self.labels=='age')[0]][0]
    a = theta[np.where(self.labels=='a')[0]][0]
    b = theta[np.where(self.labels=='b')[0]][0]
    c = theta[np.where(self.labels=='c')[0]][0]
    n = theta[np.where(self.labels=='n')[0]][0]
    
    f_BV = a*np.power(self.bmv - c, b)
    g_t = np.power(age, n)
    model = f_BV * g_t
    
    return model