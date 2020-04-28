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
        self.r12 = np.conj(self.r1)*self.r2
        self.r13 = np.conj(self.r1)*self.r3
        self.r21 = np.conj(self.r2)*self.r1
        self.r23 = np.conj(self.r2)*self.r3
        self.r31 = np.conj(self.r3)*self.r1
        self.r32 = np.conj(self.r3)*self.r2
        self.rbar = np.stack((self.r1, self.r2, self.r3), axis=2)

        ## create a data correlation matrix
        self.rmat = np.zeros((self.rbar.shape[0], self.rbar.shape[1], self.rbar.shape[2], self.rbar.shape[2]), dtype='complex')        
        
        for ii in range(self.rbar.shape[0]):
            for jj in range(self.rbar.shape[1]):
                self.rmat[ii, jj, :, :] = np.tensordot(np.conj(self.rbar[ii, jj, :]), self.rbar[ii, jj, :], axes=0 )



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
        self.theta_prior = (alpha, log_omega0, log_Np, log_Na)
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
        #S1, S2, S3 = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)        
    
        C_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na) 
        
        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(C_noise, -1, 0)

        ## take inverse and determinant
        inv_cov = np.linalg.inv(cov_mat)
        det_cov =  np.repeat(np.linalg.det(cov_mat).reshape(self.fdata.size, 1), self.r1.shape[1], axis=1)

        '''
        import time;
        t0 = time.time()
        ## vector of matrixes
        rbar = np.moveaxis(np.stack((self.r1, self.r2, self.r3), axis=2), 1, -1)

        loglike0 = 0

        ## loop over frequencies
        for ii in range(self.fdata.size):

            ## In the einstein summation j is an index over time while 
            ## m and n are indices over the elements of the covariance matrix. 

            loglike0 = loglike0 - np.real(np.einsum('mj,mn,nj', np.conj(rbar[ii, :, :]), inv_cov[ii, :, :], rbar[ii, :, :])) - \
                        rbar.shape[-1] * np.sum(np.log(det_cov[ii]))

        t1 = time.time()

        print('method 1 takes ' + str(t1-t0))

        t2 = time.time()
        '''

        inv11 = np.repeat(inv_cov[:, 0, 0].reshape(self.fdata.size, 1), self.r1.shape[1], axis=1)
        inv12 = np.repeat(inv_cov[:, 0, 1].reshape(self.fdata.size, 1), self.r1.shape[1], axis=1)
        inv13 = np.repeat(inv_cov[:, 0, 2].reshape(self.fdata.size, 1), self.r1.shape[1], axis=1)
        inv21 = np.repeat(inv_cov[:, 1, 0].reshape(self.fdata.size, 1), self.r1.shape[1], axis=1)
        inv22 = np.repeat(inv_cov[:, 1, 1].reshape(self.fdata.size, 1), self.r1.shape[1], axis=1)
        inv23 = np.repeat(inv_cov[:, 1, 2].reshape(self.fdata.size, 1), self.r1.shape[1], axis=1)
        inv31 = np.repeat(inv_cov[:, 2, 0].reshape(self.fdata.size, 1), self.r1.shape[1], axis=1)
        inv32 = np.repeat(inv_cov[:, 2, 1].reshape(self.fdata.size, 1), self.r1.shape[1], axis=1)
        inv33 = np.repeat(inv_cov[:, 2, 2].reshape(self.fdata.size, 1), self.r1.shape[1], axis=1)


        logL = -np.sum( (np.abs(self.r1)**2)*inv11 +  (np.abs(self.r2)**2)*inv22  + (np.abs(self.r3)**2)*inv33 \
                    + self.r12*inv12 + self.r13*inv13 + self.r21*inv21 + self.r23*inv23 + self.r31*inv31 \
                    + self.r32*inv32 + np.log(det_cov) )

        loglike = np.real(logL)

        return loglike




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
        cov_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)        

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        Omegaf = 10**(log_omega0)*(self.fdata/self.params['fref'])**alpha

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels. 
        cov_sgwb = Sgw[None, None, :, None]*self.response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov = np.linalg.inv(cov_mat)
        det_cov = np.linalg.det(cov_mat)
       
        logL = -np.sum(inv_cov*self.rmat) - np.sum(np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))

        loglike = np.real(logL)

        return loglike



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
        S1, S2, S3 = self.aet_noise_spectrum(self.fdata,self.f0, Np, Na) 

        
        ## Signal PSD
        H0 = 2.2*10**(-18)
        #Omegaf = 10**(log_omega0)*(self.fdata/self.params['fref'])**alpha
        Omegaf = np.tensordot(10**(log_omega0),(self.fdata/self.params['fref'])**alpha, axes=0 )

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        # Spectrum of an anisotropic SGWB signal as seen in LISA data, ie convoluted with the
        # detector response tensor. R1, R2 and R3 here are 2-d arrays over frequency and spherical 
        # harmonic coeffcients
     
        S1_gw = np.sum(Sgw.T*self.R1, axis=1)
        S2_gw = np.sum(Sgw.T*self.R2, axis=1)
        S3_gw = np.sum(Sgw.T*self.R3, axis=1)

        ## We will assume that the covariance matrix is diagonal and will only calcualte those terms. 
        ## This is true for an equal arm stationary lisa. 
        S1_net, S2_net, S3_net = S1[:, None] + S1_gw, S2[:, None] +  S2_gw, S3[:,None] + S3_gw
        
        SA_net = np.repeat(SA_net.reshape(SA_net.size, 1), self.r1.shape[1], axis=1)
        ST_net = np.repeat(ST_net.reshape(ST_net.size, 1), self.r2.shape[1], axis=1)
        SE_net = np.repeat(SE_net.reshape(SE_net.size, 1), self.r3.shape[1], axis=1)
        
        Loglike  = - np.sum( (np.abs(self.r1)**2)/S1_net + (np.abs(self.r2)**2)/S2_net  + (np.abs(self.r3)**2)/S3_net  + \
             np.log(2*np.pi*S1_net) + np.log(2*np.pi*S2_net) + np.log(2*np.pi*S3_net))


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

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels. 
        ## change axis order to make taking an inverse easier
        cov_sgwb = Sgw[:, None, None, None]*np.moveaxis(self.response_mat, [-2, -1, -3], [0, 1, 2])

        
        ## take inverse and determinant
        inv_cov = np.linalg.inv(cov_sgwb)
        det_cov = np.linalg.det(cov_sgwb)
       
        logL = -np.sum(inv_cov*self.rmat) - np.sum(np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))

        loglike = np.real(logL)

        return loglike
