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
        C_xyz = self.xyz_noise_spectrum(freqs, f0, Np, Na)

        ## Upnack xyz matrix to make assembling the aet matrix easier
        CXX, CYY, CZZ = C_xyz[0, 0], C_xyz[1, 1], C_xyz[2, 2]
        CXY, CXZ, CYZ = C_xyz[0, 1], C_xyz[0, 2], C_xyz[1, 2]


        ## construct AET matrix elements
        CAA = (1/9) * (4*CXX + CYY + CZZ - 2*CXY - 2*np.conj(CXY) - 2*CXZ - 2*np.conj(CXZ) + \
                        CYZ  + np.conj(CYZ))

        CEE = (1/3) * (CZZ + CYY - CYZ - np.conj(CYZ))

        CTT = (1/9) * (CXX + CYY + CZZ + CXY + np.conj(CXY) + CXZ + np.conj(CXZ) + CYZ + np.conj(CYZ))

        CAE = (1/(3*np.sqrt(3))) * (CYY - CZZ - CYZ + np.conj(CYZ) + 2*CXZ - 2*CXY)

        CAT = (1/9) * (2*CXX - CYY - CZZ + 2*CXY - np.conj(CXY) + 2*CXZ - np.conj(CXZ) - CYZ - np.conj(CYZ))

        CET = (1/(3*np.sqrt(3))) * (CZZ - CYY - CYZ + np.conj(CYZ) + np.conj(CXZ) - np.conj(CXY))

        C_aet = np.array([ [CAA, CAE, CAT] , \
                                    [np.conj(CAE), CEE, CET], \
                                    [np.conj(CAT), np.conj(CET), CTT] ])


        return C_aet

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
