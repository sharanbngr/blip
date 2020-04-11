from __future__ import division
import numpy as np
import types

class instrNoise():

    '''
    Class with methods to calcualte instrumental noises
    '''

    def fundamental_noise_spectrum(self, freqs, Np=4e-41, Na=1.44e-48):

        '''
        Creates a frequency array of fundamentla noise estimates for lisa. Currently we consisder only contain only
        position and acceleration noise sources. The default values are specifications pulled from 2017 Lisa proposal
        noise estimations.

        Parameters
        -----------

        freqs   : float
            A numpy array of frequencies

        Np (optional) : float
            Position noise value
        
        Na (optional) : float
            Acceleration noise level
    

        Returns
        ---------

        Sp, Sa   :   float
            Frequencies array for position and acceleration noises for each satellite
        ''' 
        
        Sp = Np*(1 + (2e-3/freqs)**4)
        Sa = Na*(1 + 16e-8/freqs**2)*(1 + (freqs/8e-3)**4)*(1.0/(2*np.pi*freqs)**4)

        return Sp, Sa

    def aet_noise_spectrum(self, freqs,f0, Np=4e-41, Na=1.44e-48):

        '''
        Calculates A, E, and T channel noise spectra for a stationary lisa. Following the defintions in
        Adams & Cornish, http://iopscience.iop.org/article/10.1088/0264-9381/18/17/308


        Parameters
        -----------

        freqs   : float
            A numpy array of frequencies

        Np (optional) : float
            Position noise value
        
        Na (optional) : float
            Acceleration noise level
    

        Returns
        ---------

        SAA, SEE, STT   :   float
            Frequencies arrays with the noise PSD for the A, E and T TDI channels


        '''

        # Get Sp and Sa
        Sp, Sa = self.fundamental_noise_spectrum(freqs, Np, Na)


        ## Noise spectra of the TDI Channels
        SAA = (16.0/3.0) * ((np.sin(2*self.f0))**2) * Sp*(np.cos(2*self.f0) + 2) \
            + (16.0/3.0) * ((np.sin(2*self.f0))**2) * Sa*(4*np.cos(2*self.f0) + 2*np.cos(4*self.f0) + 6)


        SEE = (16.0/3.0) * ((np.sin(2*self.f0))**2) * Sp*(2 + np.cos(2*self.f0)) \
            + (16.0/3.0) * ((np.sin(2*self.f0))**2) * Sa*(4 + 4*np.cos(2*self.f0) +  4*(np.cos(2*self.f0))**2 )

        STT = (16.0/3.0) * ((np.sin(2*self.f0))**2) * Sp*(1 - np.cos(2*self.f0)) \
            + (16.0/3.0) * ((np.sin(2*self.f0))**2) * Sa*(2 - 4*np.cos(2*self.f0) + 2*(np.cos(2*self.f0))**2)


        return SAA, SEE, STT


    def xyz_noise_spectrum(self, freqs,f0, Np=4e-41, Na=1.44e-48):

        '''
        Calculates X,Y,Z channel noise spectra for a stationary lisa. Following the defintions in
        Adams & Cornish, http://iopscience.iop.org/article/10.1088/0264-9381/18/17/308


        Parameters
        -----------

        freqs   : float
            A numpy array of frequencies

        Np (optional) : float
            Position noise value
        
        Na (optional) : float
            Acceleration noise level
    

        Returns
        ---------

        SAA, SEE, STT   :   float
            Frequencies arrays with the noise PSD for the A, E and T TDI channels


        '''

        C_mich = self.mich_noise_spectrum(freqs, f0, Np, Na)

        ## Noise spectra of the X, Y and Z channels
        #SX = 4*SM1* np.sin(2*f0)**2 

        C_xyz =  4 * np.sin(2*f0)**2 * C_mich 

        return C_xyz

    def mich_noise_spectrum(self, freqs,f0, Np=4e-41, Na=1.44e-48):

        '''
        Calculates michelson channel noise spectra for a stationary lisa. Following the defintions in
        Adams & Cornish, http://iopscience.iop.org/article/10.1088/0264-9381/18/17/308. We assume that
        there is no phase noise. 


        Parameters
        -----------

        freqs   : float
            A numpy array of frequencies

        Np (optional) : float
            Position noise value
        
        Na (optional) : float
            Acceleration noise level
    

        Returns
        ---------

        SAA, SEE, STT   :   float
            Frequencies arrays with the noise PSD for the A, E and T TDI channels


        '''
 
        # Get Sp and Sa
        Sp, Sa = self.fundamental_noise_spectrum(freqs, Np, Na)

     
        ## Noise spectra of the michelson channels
        S_auto  = 4.0 * (2.0 * Sa * (1.0 + (np.cos(2*f0))**2)  + Sp)
        S_cross =  (-2 * Sp - 8 * Sa) * np.cos(2*f0)

        C_mich = np.array([[S_auto, S_cross, S_cross], [S_cross, S_auto, S_cross], [S_cross, S_cross, S_auto]])

        return C_mich