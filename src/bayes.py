
from __future__ import division
import numpy as np

class Bayes():

    '''
    Class with methods for bayesian analysis of different kinds of signals. The methods currently
    include prior and likelihood functions for ISGWB analysis, sky-pixel/radiometer type analysis
    and a power spectra based spherical harmonic analysis. 
    '''

    def __init__(self):
        '''
        Init for intializing 
        '''
        aa = 1

    def instr_prior(self, theta):


        '''
        Prior function for only instrumental noise

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube. 

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref, Np and Na
    
        '''


        # Unpack: Theta is defined in the unit cube
        log_Np, log_Na = theta

        # Transform to actual priors
        log_Np = -5*log_Np - 39
        log_Na = -5*log_Na - 46

        return (log_Np, log_Na)


    def isgwb_prior(self, theta):


        '''
        Prior function for an isotropic stochastic backgound analysis.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube. 

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref, Np and Na
    
        '''


        # Unpack: Theta is defined in the unit cube
        alpha, log_omega0, log_Np, log_Na = theta

        # Transform to actual priors
        alpha       =  10*alpha-5
        log_omega0  = -10*log_omega0 - 4
        log_Np      = -5*log_Np - 39
        log_Na      = -5*log_Na - 46

        return (alpha, log_omega0, log_Np, log_Na)


    def sph_prior(self, theta):

        '''
        Prior for a power spectra based spherical harmonic anisotropic analysis

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube. 

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element is always alpha and the last two are always Np and Na
        '''



        # Prior on theta[0] which is alpha
        theta[0] = 10*theta[0] - 5

        # The rest of the priors except for the last two are ln_omegas
        for ii in range(1, theta.size-2):
            theta[ii] = -10*theta[ii] - 4

        # The last two are the priors on the position and acc noise terms. 
        theta[-2] = -5*theta[-2] - 39
        theta[-1] = -5*theta[-1] - 46
        
    
        return theta

    def instr_log_likelihood(self, theta):

        '''
        Calculate likelihood for only instrumental noise
        

        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elementes are interpreted as samples for  Np and Na respectively. 

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''


        # unpack priors
        log_Np, log_Na  = theta

        Np, Na =  10**(log_Np), 10**(log_Na)
        
        # Modelled Noise PSD
        S1, S2, S3 = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)        
    
        ## We will assume that the covariance matrix is diagonal and will only calcualte those terms. 
        ## This is true for an equal arm stationary lisa. 

        S1 = np.repeat(S1.reshape(S1.size, 1), self.r1.shape[1], axis=1)
        S2 = np.repeat(S2.reshape(S2.size, 1), self.r2.shape[1], axis=1)
        S3 = np.repeat(S3.reshape(S3.size, 1), self.r3.shape[1], axis=1)


        Loglike  = - np.sum( (np.abs(self.r1)**2)/S1 + (np.abs(self.r2)**2)/S2 + (np.abs(self.r3)**2)/S3 + \
             np.log(2*np.pi*S1) + np.log(2*np.pi*S2) + np.log(2*np.pi*S3) )
    
        return Loglike



    def isgwb_log_likelihood(self, theta):

        '''
        Calculate likelihood for an isotropic stochastic background analysis.
        

        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elementes are interpreted as samples for alpha, omega_ref, Np and Na respectively. 

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''


        # unpack priors
        alpha, log_omega0, log_Np, log_Na  = theta

        Np, Na =  10**(log_Np), 10**(log_Na)

        # Modelled Noise PSD
        SAA, SEE, STT = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)        

        ## Signal PSD
        H0 = 2.2*10**(-18)
        Omegaf = 10**(log_omega0)*(self.fdata/self.params['fref'])**alpha

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        # Spectrum of the SGWB signal as seen in LISA data, ie convoluted with the
        # detector response tensor.
        SA_gw = Sgw*self.R1
        SE_gw = Sgw*self.R2
        ST_gw = Sgw*self.R3

        ## We will assume that the covariance matrix is diagonal and will only calcualte those terms. 
        ## This is true for an equal arm stationary lisa. 
        
        SA_net, SE_net, ST_net = SAA + SA_gw, SEE +  SE_gw, STT + ST_gw
      
        SA_net = np.repeat(SA_net.reshape(SA_net.size, 1), self.r1.shape[1], axis=1)
        ST_net = np.repeat(ST_net.reshape(ST_net.size, 1), self.r2.shape[1], axis=1)
        SE_net = np.repeat(SE_net.reshape(SE_net.size, 1), self.r3.shape[1], axis=1)


        Loglike  = - 0.5*np.sum( (np.abs(self.r1)**2)/SA_net + (np.abs(self.r2)**2)/SE_net + \
             np.log(2*np.pi*SA_net) + np.log(2*np.pi*SE_net) )

        #Loglike = -np.sum( (np.abs(self.r1)**2)/SA_net +  np.log(2*np.pi*SA_net))
    
        return Loglike

    def orbiting_isgwb_log_likelihood(self, theta):

        '''
        Calculate likelihood for an isotropic stochastic background analysis.
        

        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elementes are interpreted as samples for alpha, omega_ref, Np and Na respectively. 

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''


        # unpack priors
        alpha, log_omega0, log_Np, log_Na  = theta

        Np, Na =  10**(log_Np), 10**(log_Na)

        # Modelled Noise PSD
        SAA, SEE, STT = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)        

        ## Signal PSD
        H0 = 2.2*10**(-18)
        Omegaf = 10**(log_omega0)*(self.fdata/self.params['fref'])**alpha

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        # Spectrum of the SGWB signal as seen in LISA data, ie convoluted with the
        # detector response tensor.
        SA_gw = Sgw*self.R1
        SE_gw = Sgw*self.R2
        ST_gw = Sgw*self.R3

        ## We will assume that the covariance matrix is diagonal and will only calcualte those terms. 
        ## This is true for an equal arm stationary lisa. 
        
        SA_net, SE_net, ST_net = SAA + SA_gw, SEE +  SE_gw, STT + ST_gw
      
#        SA_net = np.repeat(SA_net.reshape(SA_net.size, 1), self.r1.shape[1], axis=1)
#        ST_net = np.repeat(ST_net.reshape(ST_net.size, 1), self.r2.shape[1], axis=1)
#        SE_net = np.repeat(SE_net.reshape(SE_net.size, 1), self.r3.shape[1], axis=1)
        
        SA_net = SA_net.T
        SE_net = SE_net.T
        ST_net = ST_net.T

        Loglike  = - 0.5*np.sum( (np.abs(self.r1)**2)/SA_net + (np.abs(self.r2)**2)/SE_net + \
             np.log(2*np.pi*SA_net) + np.log(2*np.pi*SE_net) )
    
        return Loglike

    def sph_log_likelihood(self, theta):

        '''
        Calculate likelihood for a power-spectra based spherical harmonic analysis.
        

        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements are  interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element is always alpha and the last two are always Np and Na. 

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''


        # unpack priors
        alpha, log_Np, log_Na  = theta[0],theta[-2], theta[-1]
        log_omega0  = theta[1:-2]


        Np, Na =  10**(log_Np), 10**(log_Na)
        
        # Modelled Noise PSD
        SAA, SEE, STT = self.aet_noise_spectrum(self.fdata,self.f0, Np, Na) 

        
        ## Signal PSD
        H0 = 2.2*10**(-18)
        #Omegaf = 10**(log_omega0)*(self.fdata/self.params['fref'])**alpha
        Omegaf = np.tensordot(10**(log_omega0),(self.fdata/self.params['fref'])**alpha, axes=0 )

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        # Spectrum of an anisotropic SGWB signal as seen in LISA data, ie convoluted with the
        # detector response tensor. R1, R2 and R3 here are 2-d arrays over frequency and spherical 
        # harmonic coeffcients
     
        SA_gw = 0.5*np.sum(Sgw.T*self.R1, axis=1)
        SE_gw = 0.5*np.sum(Sgw.T*self.R2, axis=1)
        ST_gw = 0.5*np.sum(Sgw.T*self.R3, axis=1)

        ## We will assume that the covariance matrix is diagonal and will only calcualte those terms. 
        ## This is true for an equal arm stationary lisa. 
        SA_net, SE_net, ST_net = SAA + SA_gw, SEE +  SE_gw, STT + ST_gw

        SA_net = np.repeat(SA_net.reshape(SA_net.size, 1), self.r1.shape[1], axis=1)
        ST_net = np.repeat(ST_net.reshape(ST_net.size, 1), self.r2.shape[1], axis=1)
        SE_net = np.repeat(SE_net.reshape(SE_net.size, 1), self.r3.shape[1], axis=1)
        
        Loglike  = -np.sum( (np.abs(self.r1)**2)/SA_net + (np.abs(self.r2)**2)/SE_net + \
             np.log(2*np.pi*SA_net) + np.log(2*np.pi*SE_net) )


        if np.isnan(Loglike):
            import pdb; pdb.set_trace()
        return Loglike



    def isgwb_only_prior(self, theta):


        '''
        Prior function for an isotropic stochastic backgound analysis.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube. 

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref
    
        '''


        # Unpack: Theta is defined in the unit cube
        alpha, log_omega0  = theta

        # Transform to actual priors
        alpha       = 10*alpha-5
        log_omega0  = -10*log_omega0 - 4

        return (alpha, log_omega0)

    def isgwb_only_log_likelihood(self, theta):

        '''
        Calculate likelihood for an isotropic stochastic background analysis.
        

        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elementes are interpreted as samples for alpha, omega_ref, Np and Na respectively. 

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''


        # unpack priors
        alpha, log_omega0  = theta 

        ## Signal PSD
        H0 = 2.2*10**(-18)
        Omegaf = 10**(log_omega0)*(self.fdata/self.params['fref'])**alpha

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        # Spectrum of the SGWB signal as seen in LISA data, ie convoluted with the
        # detector response tensor.
        SA = Sgw*self.R1
        SE = Sgw*self.R2
        ST = Sgw*self.R3


      
        SA = np.repeat(SA.reshape(SA.size, 1), self.r1.shape[1], axis=1)
        ST = np.repeat(ST.reshape(ST.size, 1), self.r2.shape[1], axis=1)
        SE = np.repeat(SE.reshape(SE.size, 1), self.r3.shape[1], axis=1)

        Loglike  = - 0.5*np.sum( (np.abs(self.r1)**2)/SA + (np.abs(self.r2)**2)/SE + (np.abs(self.r3)**2)/ST + \
             np.log(2*np.pi*SA) + np.log(2*np.pi*SE) + np.log(2*np.pi*ST)  )

    
        return Loglike
