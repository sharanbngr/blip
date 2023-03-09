import numpy as np
#from line_profiler import LineProfiler

class likelihoods():

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



    def isgwb_only_log_likelihood(self, theta):

        '''
        Calculate likelihood for an isotropic stochastic background analysis.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements
            are interpreted as samples for alpha, omega_ref, Np and Na respectively.

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
        inv_cov, det_cov = bespoke_inv(cov_mat)

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))

        loglike = np.real(logL)

        return loglike


    def instr_log_likelihood(self, theta):

        '''
        Calculate likelihood for only instrumental noise


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements
            are interpreted as samples for  Np and Na respectively.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''


        # unpack priors
        log_Np, log_Na  = theta

        Np, Na =  10**(log_Np), 10**(log_Na)

        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)


        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_noise, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))


        loglike = np.real(logL)

        return loglike


    def isgwb_pl_log_likelihood(self, theta):

        '''
        Calculate likelihood for an isotropic stochastic background analysis.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements
            are interpreted as samples for alpha, omega_ref, Np and Na respectively.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''

        # unpack priors
        log_Np, log_Na, alpha, log_omega0 = theta

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
        inv_cov, det_cov = bespoke_inv(cov_mat)

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))


        loglike = np.real(logL)

        return loglike
    
    def isgwb_bpl_log_likelihood(self, theta):

        '''
        Calculate likelihood for an isotropic stochastic background analysis with a broken power law spectral model that asymptotes to alpha=2/3 (CBC) at low frequencies.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements
            are interpreted as samples for alpha, omega_ref, Np and Na respectively.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''

        # unpack priors
        log_Np, log_Na, log_A1, alpha_1, log_A2 = theta

        Np, Na =  10**(log_Np), 10**(log_Na)

        ## fixed pl (asymptotes to alpha=2/3)
        alpha_2 = alpha_1 - 0.667

        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        
        ## spectral model
        Omegaf = ((10**log_A1)*(self.fdata/self.params['fref'])**alpha_1)/(1 + (10**log_A2)*(self.fdata/self.params['fref'])**alpha_2)

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*self.response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))


        loglike = np.real(logL)

        return loglike
    
    
    def isgwb_fbpl_log_likelihood(self, theta):

        '''
        Calculate likelihood for an isotropic stochastic background analysis with a free broken power law spectral model.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements
            are interpreted as samples for alpha, omega_ref, Np and Na respectively.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''

        # unpack priors
        log_Np, log_Na, log_A1, alpha_1, log_A2, alpha_2 = theta

        Np, Na =  10**(log_Np), 10**(log_Na)


        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        
        ## spectral model
        Omegaf = ((10**log_A1)*(self.fdata/self.params['fref'])**alpha_1)/(1 + (10**log_A2)*(self.fdata/self.params['fref'])**alpha_2)

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*self.response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))


        loglike = np.real(logL)

        return loglike

    def isgwb_bpl2_log_likelihood(self, theta):

        '''
        Calculate likelihood for an isotropic stochastic background analysis with an improved, nondegenerate (or less-degenerate) broken power law spectral model.
        Assumes fixed break frequency of 1mHz and smoothing factor of 0.1.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements
            are interpreted as samples for alpha, omega_ref, Np and Na respectively.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''

        # unpack priors
        log_Np, log_Na, log_omega0, alpha_1, alpha_2, log_fbreak = theta

        Np, Na =  10**(log_Np), 10**(log_Na)

        fbreak = 10**log_fbreak
#        ## fixed pl (asymptotes to alpha=2/3)
#        alpha_2 = alpha_1 - 0.667
        
        ## fixed smoothing factor
        delta = 0.1
        
        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        
        ## spectral model
        Omegaf = (10**log_omega0)*(self.fdata/fbreak)**(alpha_1) * (0.5*(1+(self.fdata/fbreak)**(1/delta)))**((alpha_1-alpha_2)*delta)

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*self.response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))


        loglike = np.real(logL)

        return loglike


    def isgwb_tbpl_log_likelihood(self, theta):

        '''
        Calculate likelihood for an isotropic stochastic background analysis with truncated broken power law spectral model.
        Assumes fixed break frequency of 1mHz and smoothing factor of 0.1.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements
            are interpreted as samples for alpha, omega_ref, Np and Na respectively.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''
        
        # unpack priors
        log_Np, log_Na, log_omega0, alpha_1, alpha_2, log_fbreak, log_fscale = theta
        Np, Na =  10**(log_Np), 10**(log_Na)

        fbreak = 10**log_fbreak
        fcut = fbreak
        fscale = 10**log_fscale
        
        ## fixed smoothing factor
        delta = 0.1
        
        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        
        ## spectral model
        Omegaf = 0.5 * (10**log_omega0)*(self.fdata/fbreak)**(alpha_1) * (0.5*(1+(self.fdata/fbreak)**(1/delta)))**((alpha_1-alpha_2)*delta) * (1+np.tanh((fcut-self.fdata)/fscale))

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*self.response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))


        loglike = np.real(logL)

        return loglike

    def isgwb_tpl_log_likelihood(self, theta):

        '''
        Calculate likelihood for an isotropic stochastic background analysis with truncated power law spectral model.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements
            are interpreted as samples for alpha, omega_ref, Np and Na respectively.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''
        
        # unpack priors
        log_Np, log_Na, log_omega0, alpha, log_fcut, log_fscale = theta
        Np, Na =  10**(log_Np), 10**(log_Na)

        fcut = 10**log_fcut
        fscale = 10**log_fscale
        
        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        
        ## spectral model
        Omegaf = 0.5 * (10**log_omega0)*(self.fdata/self.params['fref'])**(alpha) * (1+np.tanh((fcut-self.fdata)/fscale))

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*self.response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))


        loglike = np.real(logL)

        return loglike

    def sph_pl_log_likelihood(self, theta):

        '''
        Calculate likelihood for a power-spectra based spherical harmonic analysis with a power law spectral model.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements are
            interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element
            is always alpha and the last two are always Np and Na.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''

        # unpack priors
        log_Np, log_Na, alpha, log_omega0  = theta[0],theta[1], theta[2], theta[3]

        Np, Na =  10**(log_Np), 10**(log_Na)

        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata, self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        
        ## spectral model
        Omegaf = 10**(log_omega0)*(self.fdata/self.params['fref'])**alpha

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        ## rm this line later
        # blm_theta  = np.append([0.0], theta[4:])

        blm_theta  = theta[4:]

        ## Convert the blm parameter space values to alm values.
        blm_vals = self.blm_params_2_blms(blm_theta)
        alm_vals = self.blm_2_alm(blm_vals)

        ## normalize
        alm_vals = alm_vals/(alm_vals[0] * np.sqrt(4*np.pi))

        summ_response_mat = np.einsum('ijklm,m', self.response_mat, alm_vals)

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*summ_response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)
        

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))

        loglike = np.real(logL)
        return loglike
    
    def sph_bpl_log_likelihood(self, theta):

        '''
        Calculate likelihood for a power-spectra based spherical harmonic analysis with a broken power law spectral model that asymptotes to alpha=2/3 (CBC) at low frequencies.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements are
            interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element
            is always alpha and the last two are always Np and Na.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''

        # unpack priors
        log_Np, log_Na, log_A1, alpha_1, log_A2  = theta[0],theta[1], theta[2], theta[3], theta[4]

        Np, Na =  10**(log_Np), 10**(log_Na)
        
        ## fixed pl (asymptotes to alpha=2/3)
        alpha_2 = alpha_1 - 0.667

        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata, self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        
        ## spectral model
        Omegaf = ((10**log_A1)*(self.fdata/self.params['fref'])**alpha_1)/(1 + (10**log_A2)*(self.fdata/self.params['fref'])**alpha_2)

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        ## rm this line later
        # blm_theta  = np.append([0.0], theta[4:])

        blm_theta  = theta[5:]

        ## Convert the blm parameter space values to alm values.
        blm_vals = self.blm_params_2_blms(blm_theta)
        alm_vals = self.blm_2_alm(blm_vals)

        ## normalize
        alm_vals = alm_vals/(alm_vals[0] * np.sqrt(4*np.pi))

        summ_response_mat = np.einsum('ijklm,m', self.response_mat, alm_vals)

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*summ_response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)
        

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))

        loglike = np.real(logL)
        return loglike
    

    def sph_fbpl_log_likelihood(self, theta):

        '''
        Calculate likelihood for a power-spectra based spherical harmonic analysis with a broken power law spectral model.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements are
            interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element
            is always alpha and the last two are always Np and Na.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''

        # unpack priors
        log_Np, log_Na, log_A1, alpha_1, log_A2, alpha_2  = theta[0],theta[1], theta[2], theta[3], theta[4], theta[5]

        Np, Na =  10**(log_Np), 10**(log_Na)

        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata, self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        
        ## spectral model
        Omegaf = ((10**log_A1)*(self.fdata/self.params['fref'])**alpha_1)/(1 + (10**log_A2)*(self.fdata/self.params['fref'])**alpha_2)

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2

        ## rm this line later
        # blm_theta  = np.append([0.0], theta[4:])

        blm_theta  = theta[6:]

        ## Convert the blm parameter space values to alm values.
        blm_vals = self.blm_params_2_blms(blm_theta)
        alm_vals = self.blm_2_alm(blm_vals)

        ## normalize
        alm_vals = alm_vals/(alm_vals[0] * np.sqrt(4*np.pi))

        summ_response_mat = np.einsum('ijklm,m', self.response_mat, alm_vals)

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*summ_response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)
        

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))

        loglike = np.real(logL)
        return loglike

    def sph_bpl2_log_likelihood(self, theta):

        '''
        Calculate likelihood for a power-spectra based spherical harmonic analysis with a broken power law spectral model.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements are
            interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element
            is always alpha and the last two are always Np and Na.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''

        # unpack priors
        log_Np, log_Na, log_omega0, alpha_1, alpha_2, log_fbreak = theta[0],theta[1], theta[2], theta[3], theta[4], theta[5]

        Np, Na =  10**(log_Np), 10**(log_Na)

        fbreak = 10**log_fbreak
        
        ## fixed smoothing factor
        delta = 0.1
        
        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        
        ## spectral model
        Omegaf = (10**log_omega0)*(self.fdata/fbreak)**(alpha_1) * (0.5*(1+(self.fdata/fbreak)**(1/delta)))**((alpha_1-alpha_2)*delta)

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2


        ## rm this line later
        # blm_theta  = np.append([0.0], theta[4:])

        blm_theta  = theta[6:]

        ## Convert the blm parameter space values to alm values.
        blm_vals = self.blm_params_2_blms(blm_theta)
        alm_vals = self.blm_2_alm(blm_vals)

        ## normalize
        alm_vals = alm_vals/(alm_vals[0] * np.sqrt(4*np.pi))

        summ_response_mat = np.einsum('ijklm,m', self.response_mat, alm_vals)

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*summ_response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)
        

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))

        loglike = np.real(logL)
        return loglike


    def sph_tbpl_log_likelihood(self, theta):

        '''
        Calculate likelihood for a power-spectra based spherical harmonic analysis with a truncated broken power law spectral model.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements are
            interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element
            is always alpha and the last two are always Np and Na.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''

        # unpack priors
        log_Np, log_Na, log_omega0, alpha_1, alpha_2, log_fbreak, log_fscale = theta[0],theta[1], theta[2], theta[3], theta[4], theta[5], theta[6]

        Np, Na =  10**(log_Np), 10**(log_Na)

        fbreak = 10**log_fbreak
        fcut = fbreak
        fscale = 10**log_fscale
        
        ## fixed smoothing factor
        delta = 0.1
        
        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        
        ## spectral model
        Omegaf = 0.5 * (10**log_omega0)*(self.fdata/fbreak)**(alpha_1) * (0.5*(1+(self.fdata/fbreak)**(1/delta)))**((alpha_1-alpha_2)*delta) * (1+np.tanh((fcut-self.fdata)/fscale))

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2


        ## rm this line later
        # blm_theta  = np.append([0.0], theta[4:])

        blm_theta  = theta[7:]

        ## Convert the blm parameter space values to alm values.
        blm_vals = self.blm_params_2_blms(blm_theta)
        alm_vals = self.blm_2_alm(blm_vals)

        ## normalize
        alm_vals = alm_vals/(alm_vals[0] * np.sqrt(4*np.pi))

        summ_response_mat = np.einsum('ijklm,m', self.response_mat, alm_vals)

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*summ_response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)
        

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))

        loglike = np.real(logL)
        return loglike
    
    def sph_tpl_log_likelihood(self, theta):

        '''
        Calculate likelihood for a power-spectra based spherical harmonic analysis with a truncated broken power law spectral model.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements are
            interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element
            is always alpha and the last two are always Np and Na.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''

        # unpack priors
        log_Np, log_Na, log_omega0, alpha, log_fcut, log_fscale = theta[0],theta[1], theta[2], theta[3], theta[4], theta[5]

        Np, Na =  10**(log_Np), 10**(log_Na)

        fcut = 10**log_fcut
        fscale = 10**log_fscale
        
        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        
        ## spectral model
        Omegaf = 0.5 * (10**log_omega0)*(self.fdata/self.params['fref'])**(alpha) * (1+np.tanh((fcut-self.fdata)/fscale))

        # Spectrum of the SGWB
        Sgw = Omegaf*(3/(4*self.fdata**3))*(H0/np.pi)**2


        ## rm this line later
        # blm_theta  = np.append([0.0], theta[4:])

        blm_theta  = theta[6:]

        ## Convert the blm parameter space values to alm values.
        blm_vals = self.blm_params_2_blms(blm_theta)
        alm_vals = self.blm_2_alm(blm_vals)

        ## normalize
        alm_vals = alm_vals/(alm_vals[0] * np.sqrt(4*np.pi))

        summ_response_mat = np.einsum('ijklm,m', self.response_mat, alm_vals)

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*summ_response_mat

        cov_mat = cov_sgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)
        

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))

        loglike = np.real(logL)
        return loglike
    
    def multi_log_likelihood(self, theta):

        '''
        Calculate likelihood for a power-spectra based analysis searching for an anisotropic signal in the spherical harmonic basis
        with a truncated broken power law spectral model, plus an isotropic signal with a standard power law.


        Parameters
        -----------

        theta   : float
            A list or numpy array containing rescaled samples from the unit cube. The elements are
            interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element
            is always alpha and the last two are always Np and Na.

        Returns
        ---------

        Loglike   :   float
            The log-likelihood value at the sampled point in the parameter space
        '''

        # unpack priors
        log_Np, log_Na, log_omega0_a, alpha_a, log_fcut, log_fscale, log_omega0_i, alpha_i = theta[0],theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7]

        Np, Na =  10**(log_Np), 10**(log_Na)

        fcut = 10**log_fcut
        fscale = 10**log_fscale
        
        # Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.tsegmid.size, axis=3)

        ## Signal PSD
        H0 = 2.2*10**(-18)
        
        ## anisotropic spectral model
        Omegaf_a = 0.5 * (10**log_omega0_a)*(self.fdata/self.params['fref'])**(alpha_a) * (1+np.tanh((fcut-self.fdata)/fscale))

        # Spectrum of the anisotropic SGWB
        Sgw_a = Omegaf_a*(3/(4*self.fdata**3))*(H0/np.pi)**2
        
        ## isotropic spectral model
        Omegaf_i = (10**log_omega0_i)*(self.fdata/self.params['fref'])**(alpha_i)

        # Spectrum of the anisotropic SGWB
        Sgw_i = Omegaf_i*(3/(4*self.fdata**3))*(H0/np.pi)**2

        ## rm this line later
        # blm_theta  = np.append([0.0], theta[4:])

        blm_theta  = theta[8:]

        ## Convert the blm parameter space values to alm values.
        blm_vals = self.blm_params_2_blms(blm_theta)
        alm_vals = self.blm_2_alm(blm_vals)

        ## normalize
        alm_vals = alm_vals/(alm_vals[0] * np.sqrt(4*np.pi))

        ## anisotropic response matrix, integrated across spherical harmonics
        summ_response_mat_a = np.einsum('ijklm,m', self.response_mat_a, alm_vals)
        
        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_asgwb = Sgw_a[None, None, :, None]*summ_response_mat_a

        ## isotropic response matrix
        cov_isgwb = Sgw_i[None, None, :, None]*self.response_mat_i

        cov_mat = cov_asgwb + cov_isgwb + cov_noise

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)
        

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))

        loglike = np.real(logL)
        return loglike

def bespoke_inv(A):


    """

    compute inverse without division by det; ...xv3xc3 input, or array of matrices assumed

    Credit to Eelco Hoogendoorn at stackexchange for this piece of wizardy. This is > 3 times
    faster than numpy's det and inv methods used in a fully vectorized way as of numpy 1.19.1

    https://stackoverflow.com/questions/21828202/fast-inverse-and-transpose-matrix-in-python

    """


    AI = np.empty_like(A)

    for i in range(3):
        AI[...,i,:] = np.cross(A[...,i-2,:], A[...,i-1,:])

    det = np.einsum('...i,...i->...', AI, A).mean(axis=-1)

    inv_T =  AI / det[...,None,None]

    # inverse by swapping the inverse transpose
    return np.swapaxes(inv_T, -1,-2), det



