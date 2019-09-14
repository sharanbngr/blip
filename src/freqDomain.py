from __future__ import division
import numpy as np
from scipy.special import lpmn
import types
from src.cython_func import cython_response
import healpy as hp


class freqDomain():

    '''
    Module containing methods which do various types of frequency domain calcualtions. The methods here include calculation of antenna patters for a single doppler channel, for the three michelson channels or for the AET TDI channels and calculation of noise power spectra for various channel combinations. 
    '''

    def __init__():
        import pdb; pdb.set_trace()


    def doppler_response(self, f0, theta, phi):
        
        '''
        Calculate Antenna pattern/ detector transfer function for a GW originating in the direction of (theta, phi) for the doppler channel of a stationary LISA. Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and is input as an array
        

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        phi theta  : float
            Sky position values. 
    

        Returns
        ---------

        Rplus, Rcorss   :   float
            Plus and cross antenna Patterns for the given sky direction
        '''


        ct = np.cos(theta)
        
        ## udir is just u.r, where r is the directional vector
        udir = np.sqrt(1-ct**2) * np.sin(phi + np.pi/6)

        # Initlize arrays for the detector reponse
        Rplus, Rcross = np.zeros(f0.size), np.zeros(f0.size)

        # Calculate the detector response for each frequency

        # Calculate GW transfer function for the michelson channels
        gammaU    =    1/2 * (np.sinc(f0*(1 - udir)/np.pi)*np.exp(-1j*f0*(3+udir)) + \
                            np.sinc(f0*(1 + udir)/np.pi)*np.exp(-1j*f0*(1+udir)))


        ## Michelson Channel Antenna patterns for + pol: Fplus_u = 1/2(u x u)Gamma(udir, f):eplus

        Rplus   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2) + \
                            0.5*((np.cos(phi))**2 - ct**2))*gammaU

        ## Michelson Channel Antenna patterns for x pol
        ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross

        Rcross = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi + np.pi/3))*gammaU

        return Rplus, Rcross


    def michelson_response(self, f0, theta, phi): 

        '''
        Calculate Antenna pattern/ detector transfer function for a GW originating in the direction of (theta, phi) for the three Michelson channels of a stationary LISA. Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and is input as an array
        

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        phi theta  : float
            Sky position values. 
    

        Returns
        ---------

        R1plus, R1cross, R2plus, R2cross, R3plus, R3cross   :   float
            Plus and cross antenna Patterns for the given sky direction for the three channels
        '''

        ct = np.cos(theta)

        ## udir is just u.r, where r is the directional vector
        udir = np.sqrt(1-ct**2) * np.sin(phi + np.pi/6)
        vdir = np.sqrt(1-ct**2) * np.sin(phi - np.pi/6)
        wdir = vdir - udir

        # Calculate GW transfer function for the michelson channels
        gammaU    =    1/2 * (np.sinc(f0*(1 - udir)/np.pi)*np.exp(-1j*f0*(3+udir)) + \
                         np.sinc((f0)*(1 + udir)/np.pi)*np.exp(-1j*f0*(1+udir)))

        gammaV    =    1/2 * (np.sinc(f0*(1 - vdir)/np.pi)*np.exp(-1j*f0*(3+vdir)) + \
                         np.sinc((f0)*(1 + vdir)/np.pi)*np.exp(-1j*f0*(1+vdir)))

        gammaW    =    1/2 * (np.sinc(f0*(1 - wdir)/np.pi)*np.exp(-1j*f0*(3+wdir)) + \
                         np.sinc((f0)*(1 + wdir)/np.pi)*np.exp(-1j*f0*(1+wdir)))

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
        Calculate Antenna pattern/ detector transfer function for a GW originating in the direction of (theta, phi) for the A, E and T TDI channels of a stationary LISA. Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and is input as an array
        

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        phi theta  : float
            Sky position values. 
    

        Returns
        ---------

        RAplus, RAcross, REplus, REcross, RTplus, RTcross   :   float
            Plus and cross antenna Patterns for the given sky direction for the three channels
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
    

    def isgwb_mich_response(self, f0):

        '''
        Calcualte the Antenna pattern/ detector transfer function functions to an isotropic SGWB using basic michelson
        channels. Note that since this is the response to an isotropic background, the response function is integrated
        over sky direction and averaged over polarozation. The angular integral is a linear and rectangular in the
        cos(theta) and phi space.  Note also that f0 is (pi*L*f)/c and is input as an array

        

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

    

        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarization.
        '''

        tt = np.linspace(-1, 1, 200)
        pp = np.linspace(0, 2*np.pi, 200, endpoint=False)

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
            gammaU_plus    =    1/2 * (np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(3+udir)) + \
                             np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(1+udir)))

            gammaV_plus    =    1/2 * (np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(3+vdir)) + \
                             np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(1+vdir)))

            gammaW_plus    =    1/2 * (np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(3+wdir)) + \
                             np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(1+wdir)))
            
            
            # Calculate GW transfer function for the michelson channels
            gammaU_minus    =    1/2 * (np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(3 - udir)) + \
                             np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(1 - udir)))

            gammaV_minus    =    1/2 * (np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(3 - vdir)) + \
                             np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(1 - vdir)))

            gammaW_minus    =    1/2 * (np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(3 - wdir)) + \
                             np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(1 - wdir)))
            

            ## response function u x u : eplus
            ##  Fplus_u = (u x u):eplus

            Fplus_u   = (1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - \
                             np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2)  + \
                                 0.5*((np.cos(phi))**2 - ct**2))
        
            Fplus_v   = (1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 + \
                             np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2) + \
                                 0.5*((np.cos(phi))**2 - ct**2))

            Fplus_w   = (1 - (1+ct**2)*(np.cos(phi))**2)

            ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross
            Fcross_u  = - ct * (np.sin(2*phi + np.pi/3))
            Fcross_v  = - ct * (np.sin(2*phi - np.pi/3))
            Fcross_w   = ct*np.sin(2*phi)


            ## Michelson antenna patterns
            ## Calculate Fplus
            Fplus1 = 0.5*(Fplus_u*gammaU_plus - Fplus_v*gammaV_plus)
            Fplus2 = 0.5*(Fplus_w*gammaW_plus - Fplus_u*gammaU_minus)
            Fplus3 = 0.5*(Fplus_v*gammaV_minus - Fplus_w*gammaW_minus)

            ## Calculate Fcross
            Fcross1 = 0.5*(Fcross_u*gammaU_plus - Fcross_v*gammaV_plus)
            Fcross2 = 0.5*(Fcross_w*gammaW_plus - Fcross_u*gammaU_minus)
            Fcross3 = 0.5*(Fcross_v*gammaV_minus - Fcross_w*gammaW_minus)


            ## Detector response summed over polarization and integrated over sky direction
            R1[ii] = dct*dphi/(8*np.pi)*np.sum( (np.absolute(Fplus1))**2 + (np.absolute(Fcross1))**2 )
            R2[ii] = dct*dphi/(8*np.pi)*np.sum( (np.absolute(Fplus2))**2 + (np.absolute(Fcross2))**2 )
            R3[ii] = dct*dphi/(8*np.pi)*np.sum( (np.absolute(Fplus3))**2 + (np.absolute(Fcross3))**2 )

        return R1, R2, R3




    def isgwb_xyz_response(self, f0):

        '''
        Calcualte the Antenna pattern/ detector transfer function functions to an isotropic SGWB using X, Y and Z TDI
        channels. Note that since this is the response to an isotropic background, the response function is integrated
        over sky direction and averaged over polarozation. The angular integral is a linear and rectangular in the
        cos(theta) and phi space.  Note also that f0 is (pi*L*f)/c and is input as an array

        

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

    

        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarization.
        '''

        tt = np.linspace(-1, 1, 200)
        pp = np.linspace(0, 2*np.pi, 200, endpoint=False)

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
            gammaU_plus    =    1/2 * (np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(3+udir)) + \
                             np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(1+udir)))

            gammaV_plus    =    1/2 * (np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(3+vdir)) + \
                             np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(1+vdir)))

            gammaW_plus    =    1/2 * (np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(3+wdir)) + \
                             np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(1+wdir)))
            
            
            # Calculate GW transfer function for the michelson channels
            gammaU_minus    =    1/2 * (np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(3 - udir)) + \
                             np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(1 - udir)))

            gammaV_minus    =    1/2 * (np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(3 - vdir)) + \
                             np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(1 - vdir)))

            gammaW_minus    =    1/2 * (np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(3 - wdir)) + \
                             np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(1 - wdir)))
            

            ## response function u x u : eplus
            ##  Fplus_u = (u x u):eplus

            Fplus_u   = (1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - \
                             np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2)) #  + \
                             #    0.5*((np.cos(phi))**2 - ct**2))
        
            Fplus_v   = (1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 + \
                             np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2)) # + \
                             #    0.5*((np.cos(phi))**2 - ct**2))

            Fplus_w   = (1 - (1+ct**2)*(np.cos(phi))**2)

            ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross
            Fcross_u  = - ct * (np.sin(2*phi + np.pi/3))
            Fcross_v  = - ct * (np.sin(2*phi - np.pi/3))
            Fcross_w   = ct*np.sin(2*phi)

            ## Calculate Fplus
            Fplus1 = 0.5*(Fplus_u*gammaU_plus - Fplus_v*gammaV_plus)
            Fplus2 = 0.5*(Fplus_w*gammaW_plus - Fplus_u*gammaU_minus)
            Fplus3 = 0.5*(Fplus_v*gammaV_minus - Fplus_w*gammaW_minus)

            ## Calculate Fcross
            Fcross1 = 0.5*(Fcross_u*gammaU_plus - Fcross_v*gammaV_plus)
            Fcross2 = 0.5*(Fcross_w*gammaW_plus - Fcross_u*gammaU_minus)
            Fcross3 = 0.5*(Fcross_v*gammaV_minus - Fcross_w*gammaW_minus)

            ## Calculate antenna patterns for the X, Y, Z channels.
            FXplus = 2*np.sin(2*f0[ii])*Fplus1
            FYplus = 2*np.sin(2*f0[ii])*Fplus2
            FZplus = 2*np.sin(2*f0[ii])*Fplus3

            FXcross = 2*np.sin(2*f0[ii])*Fcross1
            FYcross = 2*np.sin(2*f0[ii])*Fcross2
            FZcross = 2*np.sin(2*f0[ii])*Fcross3

            ## Detector response for the TDI Channels, summed over polarization
            ## and integrated over sky direction
            R1[ii] = dct*dphi/(8*np.pi)*np.sum( (np.absolute(FXplus))**2 + (np.absolute(FXcross))**2 )
            R2[ii] = dct*dphi/(8*np.pi)*np.sum( (np.absolute(FYplus))**2 + (np.absolute(FYcross))**2 )
            R3[ii] = dct*dphi/(8*np.pi)*np.sum( (np.absolute(FZplus))**2 + (np.absolute(FZcross))**2 )

        return R1, R2, R3



    def isgwb_aet_response(self, f0):

        '''
        Calcualte the Antenna pattern/ detector transfer function functions to an isotropic SGWB using A, E and T TDI channels. 
        Note that since this is the response to an isotropic background, the response function is integrated over sky direction
        and averaged over polarozation. The angular integral is a linear and rectangular in the cos(theta) and phi space.  Note
        that f0 is (pi*L*f)/c and is input as an array

        

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

    

        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarization.
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
            gammaU    =    1/2 * (np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(3+udir)) + \
                             np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(1+udir)))

            gammaV    =    1/2 * (np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(3+vdir)) + \
                             np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(1+vdir)))

            gammaW    =    1/2 * (np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(3+wdir)) + \
                             np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(1+wdir)))

            ##  Fplus_u = 1/2(u x u)Gamma(udir, f):eplus

            Fplus_u   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2))*gammaU #+ \
                        #    0.5*((np.cos(phi))**2 - ct**2))*gammaU

            Fplus_v   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 + np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2))*gammaV #+ \
                        #     0.5*((np.cos(phi))**2 - ct**2))*gammaV

            Fplus_w   = 1/2*(1 - (1+ct**2)*(np.cos(phi))**2)*gammaW

            ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross
            Fcross_u  = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi + np.pi/3))*gammaU
            Fcross_v  = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi - np.pi/3))*gammaV
            Fcross_w   = 1/2*ct*np.sin(2*phi)*gammaW

            ## Calculate Fplus
            Fplus1 = (Fplus_u - Fplus_v)
            Fplus2 = (Fplus_w - Fplus_u)
            Fplus3 = (Fplus_v - Fplus_w)

            ## Calculate Fcross
            Fcross1 = (Fcross_u - Fcross_v)
            Fcross2 = (Fcross_w - Fcross_u)
            Fcross3 = (Fcross_v - Fcross_w)

            ## Calculate antenna patterns for the A, E and T channels.
            FAplus = (1/3)*np.sin(2*f0[ii])*(2*Fplus1 - Fplus2 - Fplus3)
            FEplus = (1/np.sqrt(3))*np.sin(2*f0[ii])*(Fplus3 - Fplus2)
            FTplus = (1/3)*np.sin(2*f0[ii])*(Fplus1 + Fplus3 + Fplus2)

            FAcross = (1/3)*np.sin(2*f0[ii])*(2*Fcross1 - Fcross2 - Fcross3)
            FEcross = (1/np.sqrt(3))*np.sin(2*f0[ii])*(Fcross3 - Fcross2)
            FTcross = (1/3)*np.sin(2*f0[ii])*(Fcross1 + Fcross3 + Fcross2)

            ## Detector response for the TDI Channels, summed over polarization
            ## and integrated over sky direction
            R1[ii] = dct*dphi/(8*np.pi)*np.sum((np.absolute(FAplus))**2 + (np.absolute(FAcross))**2)
            R2[ii] = dct*dphi/(8*np.pi)*np.sum((np.absolute(FEplus))**2 + (np.absolute(FEcross))**2)
            R3[ii] = dct*dphi/(8*np.pi)*np.sum((np.absolute(FTplus))**2 + (np.absolute(FTcross))**2)



        return R1, R2, R3

    def asgwb_aet_response(self, f0):

        '''
        Calculate the Antenna pattern/ detector transfer function functions to acSGWB using A, E and T TDI channels,
        and using a spherical harmonic decomposition. Note that the response function is integrated over sky direction
        with the appropriate legandre polynomial, and averaged over polarozation. Finally note that the spherical harmonic
        coeffcients correspond to strain sky distribution, while the legandre polynomials describe the power sky. The 
        angular integral is a linear and rectangular in the cos(theta) and phi space.  Note that f0 is (pi*L*f)/c and is input as an array

        

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

    

        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged
            over polarization. The arrays are 2-d, one direction corresponds to frequency and the other to the l coeffcient. 
        '''

        
        tt = np.arange(-1, 1, 0.02)
        pp = np.arange(0, 2*np.pi, np.pi/100)

        [ct, phi] = np.meshgrid(tt,pp)
        dct = ct[0, 1] - ct[0,0]
        dphi = phi[1,0] - phi[0,0]

        ## udir is just u.r, where r is the directional vector
        udir = np.sqrt(1-ct**2) * np.sin(phi + np.pi/6)
        vdir = np.sqrt(1-ct**2) * np.sin(phi - np.pi/6)
        wdir = vdir - udir

        # Initlize arrays for the detector reponse
        R1 = np.zeros((f0.size, self.params['lmax'] +1))
        R2 = np.zeros((f0.size, self.params['lmax'] +1))
        R3 = np.zeros((f0.size, self.params['lmax'] +1))

        ## initalize array for plms
        plms = np.zeros((tt.size, self.params['lmax']+1, self.params['lmax'] +1 ))


        ## Get associated legandre polynomials.
        for ii in range(tt.size):
            plms[ii, :, :], _ = lpmn(self.params['lmax'], self.params['lmax'], tt[ii]) 

        ## It is the squares of the polynomials which are relevent. 
        plms = plms**2
        # Calculate the detector response for each frequency
        for ii in range(0, f0.size):

            # Calculate GW transfer function for the michelson channels
            gammaU    =    1/2 * (np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(3+udir)) + \
                             np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(1+udir)))

            gammaV    =    1/2 * (np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(3+vdir)) + \
                             np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(1+vdir)))

            gammaW    =    1/2 * (np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(3+wdir)) + \
                             np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(1+wdir)))

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
            
            R1[ii, :] = dct*dphi/(4*np.pi)*np.sum(np.tensordot((np.absolute(FAplus))**2 + \
                    (np.absolute(FAcross))**2, plms, axes=1), axis=(0, 1))
            R2[ii, :] = dct*dphi/(4*np.pi)*np.sum(np.tensordot((np.absolute(FEplus))**2 + \
                    (np.absolute(FEcross))**2, plms, axes=1), axis=(0, 1))
            R3[ii, :] = dct*dphi/(4*np.pi)*np.sum(np.tensordot((np.absolute(FTplus))**2 + \
                    (np.absolute(FTcross))**2, plms, axes=1), axis=(0,1))   



        return R1, R2, R3


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

        SM1, SM2, SM3 = self.mich_noise_spectrum(freqs, f0, Np, Na)

        ## Noise spectra of the X, Y and Z channels
        SX = 4*SM1* np.sin(2*f0)**2 


        return SX, SX, SX

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

     
        ## Noise spectra of the X, Y and Z channels
        SX = 4.0 * (2.0 * (1.0 + (np.cos(2*f0))**2) * Sa + Sp)
        

        return SX, SX, SX



    def isgwb_mich_strain_response(self, f0):

        '''
        Calculate the detector transfer function functions to an isotropic SGWB non-polarized using basic michelson
        channels. Note that since this is the response to an isotropic background, the response function is integrated
        over sky direction and averaged over polarozation. The angular integral is a linear and rectangular in the
        cos(theta) and phi space.  Note also that f0 is (pi*L*f)/c and is input as an array. The response function is given
        for the strain of the signal rather than the power

        

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

    

        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarization.
        '''

        # Define nside and npix for the healpix array
        nside = 10

        npix = hp.nside2npix(nside)

        # Array of pixel indices
        pix_idx  = np.arange(npix)

        #Angular coordinates of pixel indcides 
        theta, phi = hp.pix2ang(nside, pix_idx)

        # Take cosine. 
        ctheta = np.cos(theta)

        # Area of each pixel in sq.radians
        dOmega = hp.pixelfunc.nside2pixarea(nside)

        ## Create directional vectors
        udir = np.sqrt(1-ctheta**2) * np.sin(phi + np.pi/6)
        vdir = np.sqrt(1-ctheta**2) * np.sin(phi - np.pi/6)
        wdir = vdir - udir

        # Initlize arrays for the detector reponse
        R1 = np.zeros((f0.size, 2), dtype='complex')
        R2 = np.zeros((f0.size, 2), dtype='complex')
        R3 = np.zeros((f0.size, 2), dtype='complex')

        # Assign random complex amplitudes for each pixel
        rand_plus  = np.random.standard_normal(size=(npix, f0.size)) + 1j*np.random.standard_normal(size=(npix, f0.size))
        rand_cross = np.random.standard_normal(size=(npix, f0.size)) + 1j*np.random.standard_normal(size=(npix, f0.size))

        # Calculate the detector response for each frequency
        for ii in range(0, f0.size):

            # Calculate GW transfer function for the michelson channels
            gammaU_plus    =    1/2 * (np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(3+udir)) + \
                             np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(1+udir)))

            gammaV_plus    =    1/2 * (np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(3+vdir)) + \
                             np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(1+vdir)))

            gammaW_plus    =    1/2 * (np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(3+wdir)) + \
                             np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(1+wdir)))
            
            
            # Calculate GW transfer function for the michelson channels
            gammaU_minus    =    1/2 * (np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(3 - udir)) + \
                             np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(1 - udir)))

            gammaV_minus    =    1/2 * (np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(3 - vdir)) + \
                             np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(1 - vdir)))

            gammaW_minus    =    1/2 * (np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(3 - wdir)) + \
                             np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(1 - wdir)))
            

            ## response function u x u : eplus
            ##  Fplus_u = (u x u):eplus

            Fplus_u   = (1/4*(1-ctheta**2) + 1/2*(ctheta**2)*(np.cos(phi))**2 - \
                             np.sqrt(3/16)*np.sin(2*phi)*(1+ctheta**2)  + \
                                 0.5*((np.cos(phi))**2 - ctheta**2))
        
            Fplus_v   = (1/4*(1-ctheta**2) + 1/2*(ctheta**2)*(np.cos(phi))**2 + \
                             np.sqrt(3/16)*np.sin(2*phi)*(1+ctheta**2) + \
                                 0.5*((np.cos(phi))**2 - ctheta**2))

            Fplus_w   = (1 - (1+ctheta**2)*(np.cos(phi))**2)

            ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross
            Fcross_u  = - ctheta * (np.sin(2*phi + np.pi/3))
            Fcross_v  = - ctheta * (np.sin(2*phi - np.pi/3))
            Fcross_w   = ctheta*np.sin(2*phi)


            ## Michelson antenna patterns
            ## Calculate Fplus
            Fplus1 = 0.5*(Fplus_u*gammaU_plus - Fplus_v*gammaV_plus)
            Fplus2 = 0.5*(Fplus_w*gammaW_plus - Fplus_u*gammaU_minus)*np.exp(2j*f0[ii]*udir)
            Fplus3 = 0.5*(Fplus_v*gammaV_minus - Fplus_w*gammaW_minus)*np.exp(2j*f0[ii]*vdir)

            ## Calculate Fcross
            Fcross1 = 0.5*(Fcross_u*gammaU_plus - Fcross_v*gammaV_plus)
            Fcross2 = 0.5*(Fcross_w*gammaW_plus - Fcross_u*gammaU_minus)*np.exp(2j*f0[ii]*udir)
            Fcross3 = 0.5*(Fcross_v*gammaV_minus - Fcross_w*gammaW_minus)*np.exp(2j*f0[ii]*vdir)

            npix = Fplus1.size

            ## Detector response summed over polarization and integrated over sky direction
            R1[ii,0], R1[ii, 1] = np.sqrt(0.5/npix)*np.sum(Fplus1*rand_plus[:, ii]), np.sqrt(0.5/npix)*np.sum(Fcross1*rand_cross[:, ii]) 
            R2[ii,0], R2[ii, 1] = np.sqrt(0.5/npix)*np.sum(Fplus2*rand_plus[:, ii]), np.sqrt(0.5/npix)*np.sum(Fcross2*rand_cross[:, ii]) 
            R3[ii,0], R3[ii, 1] = np.sqrt(0.5/npix)*np.sum(Fplus3*rand_plus[:, ii]), np.sqrt(0.5/npix)*np.sum(Fcross3*rand_cross[:, ii]) 
        

        return R1, R2, R3

