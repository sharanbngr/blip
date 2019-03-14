from __future__ import division
import numpy as np

class bayes():

    '''
    Class with methods for bayesian analysis of different kinds of signals. The methods currently
    include prior and likelihood functions for ISGWB analysis, sky-pixel/radiometer type analysis
    and a power spectra based spherical harmonic analysis. 
    '''


    def isgwb_prior(self, theta):

        '''
        Prior function for the ISGWB
        '''

        # Unpack: Theta is defined in the unit cube
        alpha, log_omega0, log_Np, log_Na = theta

        # Transform to actual priors
        alpha       = 10*alpha-5
        log_omega0   = 10*log_omega0 -14
        log_Np = 5*log_Np - 44
        log_Na = 5*log_Na - 51

        return (alpha, log_omega0, log_Np, log_Na)


    def Sph_prior(self, theta):

        '''
        Prior for an power spectra spherical harmonic anisotropic analysis
        '''

        # Prior on theta[0] which is alpha
        theta[0]       = 10*theta[0] -5

        # The rest of the priors except for the last two are ln_omegas
        for ii in range(1, theta.size-2):
            theta[ii] = 10*theta[ii] - 14

        # The last two are the priors on the position and acc noise terms. 
        theta[ii-2] = 5*theta[ii-2] - 44
        theta[ii-1] = 5*theta[ii-1] - 51

        return theta

    def isgwb_log_likelihood(self, theta):

        '''
        Calculate isotropic likelihood for the sampled point theta.
        '''

        # unpack priors
        alpha, log_omega0, log_Np, log_Na  = theta

        Np, Na =  10**(log_Np), 10**(log_Na)

        # Modelled Noise PSD
        SAA, SEE, STT = self.aet_noise_spectrum(self.freqs, Np, Na, self.f0)        


        ## Signal PSD
        H0 = 2.2*10**(-18)
        Omegaf = 10**(log_omega0)*(self.freqs/self.fref)**alpha

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.freqs**3))*(H0/np.pi)**2

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

        Loglike  = -0.5*np.sum( (np.abs(self.r1)**2)/SA_net + (np.abs(self.r2)**2)/SE_net + \
             np.log(2*np.pi*SA_net) + np.log(2*np.pi*SE_net) )

        return Loglike


    def Sph_likelihood(self, theta):

        '''
        Likelihood wrapper for a power spectra based sphercial harmonic analysis. 
        '''
