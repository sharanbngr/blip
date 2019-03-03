from __future__ import division
import numpy as np

class Antennapatterns():

    '''
    Class  of the LISA array or of parts of it. The methods here include calculation of antenna patters for a single doppler channel, for
    the three michelson channels or for the AET TDI channels
    '''

    def __init__(self, params, inj):
        self.params = params
        self.inj = inj

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
'''
    def tdi_isgwb_response(self, f0): 

        tt = np.arange(-1, 1, 0.01)
        pp = np.arange(0, 2*np.pi, np.pi/100)

        for ffs in f0:

            
            ## Detector response for the TDI Channels, summed over polarization
            ## and integrated over sky direction
            R1 = dct*dphi/(4*np.pi)*np.sum((np.absolute(FAplus))**2 + (np.absolute(FAcross))**2)
            R2 = dct*dphi/(4*np.pi)*np.sum((np.absolute(FEplus))**2 + (np.absolute(FEcross))**2)
            R3 = dct*dphi/(4*np.pi)*np.sum((np.absolute(FTplus))**2 + (np.absolute(FTcross))**2)


        return R1, R2, R3
'''




