from __future__ import division
import numpy as np

class freqDomain():

    '''
    Class for various types of frequency domain calcualtions. The methods here include calculation of antenna patters for a single doppler channel, for the three michelson channels or for the AET TDI channels and calculation of noise power spectra for various channel combinations. 
    '''

    def __init__(self, params, inj):
        

    def doppler_response(self, f0, theta, phi):
        
        '''
        Calculate antenna patterns given theta and phi for the doppler channel of a stationary LISA. 
        Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and should be
        passed as an array
        '''


        ct = np.cos(theta)
        
        ## udir is just u.r, where r is the directional vector
        udir = np.sqrt(1-ct**2) * np.sin(phi + np.pi/6)

        # Initlize arrays for the detector reponse
        Rplus, Rcross = np.zeros(f0.size), np.zeros(f0.size)

        # Calculate the detector response for each frequency

        # Calculate GW transfer function for the michelson channels
        gammaU    =    1/2 * (np.sinc(f0*(1 - udir))*np.exp(-1j*f0*(3+udir)) + \
                            np.sinc(f0*(1 + udir))*np.exp(-1j*f0*(1+udir)))


        ## Michelson Channel Antenna patterns for + pol: Fplus_u = 1/2(u x u)Gamma(udir, f):eplus

        Rplus   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2) + \
                            0.5*((np.cos(phi))**2 - ct**2))*gammaU

        ## Michelson Channel Antenna patterns for x pol
        ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross

        Rcross = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi + np.pi/3))*gammaU

        return Rplus, Rcross


    def michelson_response(self, f0, theta, phi): 

        '''
        Calculate antenna patterns given theta and phi for the three michelson of a stationary LISA. 
        Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and should be
        passed as an array
        '''

        ct = np.cos(theta)

        ## udir is just u.r, where r is the directional vector
        udir = np.sqrt(1-ct**2) * np.sin(phi + np.pi/6)
        vdir = np.sqrt(1-ct**2) * np.sin(phi - np.pi/6)
        wdir = vdir - udir

        # Calculate GW transfer function for the michelson channels
        gammaU    =    1/2 * (np.sinc(f0*(1 - udir))*np.exp(-1j*f0*(3+udir)) + \
                         np.sinc((f0)*(1 + udir))*np.exp(-1j*f0*(1+udir)))

        gammaV    =    1/2 * (np.sinc(f0*(1 - vdir))*np.exp(-1j*f0*(3+vdir)) + \
                         np.sinc((f0)*(1 + vdir))*np.exp(-1j*f0*(1+vdir)))

        gammaW    =    1/2 * (np.sinc(f0*(1 - wdir))*np.exp(-1j*f0*(3+wdir)) + \
                         np.sinc((f0)*(1 + wdir))*np.exp(-1j*f0*(1+wdir)))

        ## Michelson Channel Antenna patterns for + pol
        ##  Fplus_u = 1/2(u x u)Gamma(udir, f):eplus
        Fplus_u   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2) + \
                        0.5*((np.cos(phi))**2 - ct**2))*gammaU

        Fplus_v   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 + np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2)+ \
                     0.5*((np.cos(phi))**2 - ct**2))*gammaV

        Fplus_w   = 1/2*(1 - (1+ct**2)*(np.cos(phi))**2)*gammaW

        ## Michelson Channel Antenna patterns for x pol
        ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross

        Fcross_u  = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi + np.pi/3))*gammaU
        Fcross_v  = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi - np.pi/3))*gammaV
        Fcross_w   = 1/2*ct*np.sin(2*phi)*gammaW

        ## Compelte Michelson antenna patterns
        ## Calculate Fplus
        R1plus = (Fplus_u - Fplus_v)
        R2plus = (Fplus_w - Fplus_u)
        R3plus = (Fplus_v - Fplus_w)

        ## Calculate Fcross
        R1cross = (Fcross_u - Fcross_v)
        R2cross = (Fcross_w - Fcross_u)
        R3cross = (Fcross_v - Fcross_w)

        return R1plus, R1cross, R2plus, R2cross, R3plus, R3cross

    def aet_response(self, f0, theta, phi): 

        '''
        Calculate antenna patterns given theta and phi for the A,E and T channels of a stationary LISA. 
        Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and should be
        passed as an array
        '''

        R1plus, R1cross, R2plus, R2cross, R3plus, R3cross  = self.michelson_response(f0, theta, phi)
        

        ## Calculate antenna patterns for the A, E and T channels
        RAplus = (2/3)*np.sin(2*f0)*(2*R1plus - R2plus - R3plus)
        REplus = (2/np.sqrt(3))*np.sin(2*f0)*(R3plus - R2plus)
        RTplus = (1/3)*np.sin(2*f0)*(R1plus + R3plus + R2plus)

        RAcross = (2/3)*np.sin(2*f0)*(2*R1cross - R2cross - R3cross)
        REcross = (2/np.sqrt(3))*np.sin(2*f0)*(R3cross - R2cross)
        RTcross = (1/3)*np.sin(2*f0)*(R1cross + R3cross + R2cross)

        return RAplus, RAcross, REplus, REcross, RTplus, RTcross

    def tdi_isgwb_response(self, f0): 

        '''
        Calcualte the detector response functions to an isotropic SGWB using A, E and T TDI channels. cos(theta) space and phi space. The angular integral is a linear and 
        rectangular in the cos(theta) and phi space
        '''
        
        
        tt = np.arange(-1, 1, 0.01)
        pp = np.arange(0, 2*np.pi, np.pi/100)

        [ct, phi] = np.meshgrid(tt,pp)
        dct = ct[0, 1] - ct[0,0]
        dphi = phi[1,0] - phi[0,0]

        ## udir is just u.r, where r is the directional vector
        udir = np.sqrt(1-ct**2) * np.sin(phi + np.pi/6)
        vdir = np.sqrt(1-ct**2) * np.sin(phi - np.pi/6)
        wdir = vdir - udir

        # Initlize arrays for the detector reponse
        R1 = np.zeros(f0.size)
        R2 = np.zeros(f0.size)
        R3 = np.zeros(f0.size)

        # Calculate the detector response for each frequency
        for ii in range(0, f0.size):

            # Calculate GW transfer function for the michelson channels
            gammaU    =    1/2 * (np.sinc((f0[ii])*(1 - udir))*np.exp(-1j*f0[ii]*(3+udir)) + \
                             np.sinc((f0[ii])*(1 + udir))*np.exp(-1j*f0[ii]*(1+udir)))

            gammaV    =    1/2 * (np.sinc((f0[ii])*(1 - vdir))*np.exp(-1j*f0[ii]*(3+vdir)) + \
                             np.sinc((f0[ii])*(1 + vdir))*np.exp(-1j*f0[ii]*(1+vdir)))

            gammaW    =    1/2 * (np.sinc((f0[ii])*(1 - wdir))*np.exp(-1j*f0[ii]*(3+wdir)) + \
                             np.sinc((f0[ii])*(1 + wdir))*np.exp(-1j*f0[ii]*(1+wdir)))

            ## Michelson Channel Antenna patterns for + pol
            ##  Fplus_u = 1/2(u x u)Gamma(udir, f):eplus

            Fplus_u   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2) + \
                            0.5*((np.cos(phi))**2 - ct**2))*gammaU

            Fplus_v   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 + np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2)+ \
                         0.5*((np.cos(phi))**2 - ct**2))*gammaV

            Fplus_w   = 1/2*(1 - (1+ct**2)*(np.cos(phi))**2)*gammaW


            ## Michelson Channel Antenna patterns for x pol
            ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross
            Fcross_u  = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi + np.pi/3))*gammaU
            Fcross_v  = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi - np.pi/3))*gammaV
            Fcross_w   = 1/2*ct*np.sin(2*phi)*gammaW


            ## First Michelson antenna patterns
            ## Calculate Fplus
            Fplus1 = (Fplus_u - Fplus_v)
            Fplus2 = (Fplus_w - Fplus_u)
            Fplus3 = (Fplus_v - Fplus_w)

            ## Calculate Fcross
            Fcross1 = (Fcross_u - Fcross_v)
            Fcross2 = (Fcross_w - Fcross_u)
            Fcross3 = (Fcross_v - Fcross_w)

            ## Calculate antenna patterns for the A, E and T channels -  We are switiching to doppler channel.
            FAplus = (1/3)*np.sin(2*f0[ii])*(2*Fplus1 - Fplus2 - Fplus3)
            FEplus = (1/np.sqrt(3))*np.sin(2*f0[ii])*(Fplus3 - Fplus2)
            FTplus = (1/3)*np.sin(2*f0[ii])*(Fplus1 + Fplus3 + Fplus2)

            FAcross = (1/3)*np.sin(2*f0[ii])*(2*Fcross1 - Fcross2 - Fcross3)
            FEcross = (1/np.sqrt(3))*np.sin(2*f0[ii])*(Fcross3 - Fcross2)
            FTcross = (1/3)*np.sin(2*f0[ii])*(Fcross1 + Fcross3 + Fcross2)

            ## Detector response for the TDI Channels, summed over polarization
            ## and integrated over sky direction
            R1[ii] = dct*dphi/(4*np.pi)*np.sum((np.absolute(FAplus))**2 + (np.absolute(FAcross))**2)
            R2[ii] = dct*dphi/(4*np.pi)*np.sum((np.absolute(FEplus))**2 + (np.absolute(FEcross))**2)
            R3[ii] = dct*dphi/(4*np.pi)*np.sum((np.absolute(FTplus))**2 + (np.absolute(FTcross))**2)



        return R1, R2, R3

    def fundamental_noise_spectrum(self, freqs, Np=4e-41, Na=1.44e-48):

        '''
        Fundamentla noise estikmates for lisa. Currently only contain only position and acceleration noise sources.
        The default values are specifications pulled from 2017 Lisa proposal noise estimations.
        ''' 
        
        Sp = Np
        Sa = Na*(1 + 16e-8/freqs**2)*(1.0/(2*np.pi*freqs)**4)

        return Sp, Sa

    def aet_noise_spectrum(self, freqs, Np=4e-41, Na=1.44e-48):

        '''
        A, E, and T channel noise spectra for a stationary lisa. Following the defintions in
        Adams & Cornish, http://iopscience.iop.org/article/10.1088/0264-9381/18/17/308
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


        retrun SAA, SEE, STT


