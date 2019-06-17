from __future__ import division
import numpy as np
from scipy.special import lpmn

class MovingfreqDomain():

    '''
    Module containing methods which do various types of frequency domain calcualtions. The methods here include calculation of antenna patters for a single doppler channel, for the three michelson channels or for the AET TDI channels and calculation of noise power spectra for various channel combinations. All methods are calculated for a moving LISA constellation given a particular set of satellite orbits.
    '''
    def LISA_orbits(self,timearray)
        '''
        Define LISA orbits as a function of predefined time array using analytic MLDC orbits.
        
        Parameters
        -----------
        
        timearray  :  array
            A numpy array of times in seconds.
            
        Returns
        -----------
        r1, r2, r3  :  array
            Arrays of satellite positions for reach time in timearray. e.g. r1[1] is [x1,y1,z1] at t=timearray[1]. 
        '''
        ## Semimajor axis in m
        a = 1.496e11
        ## LISA arm length in m
        L = 1.5e9
        sats = np.array([1,2,3])
        ## Alpha and beta phases allow for changing of initial satellite orbital phases; default initial conditions are alphaphase=betaphase=0.
        betaphase = 0
        alphaphase = 0
        ## Orbital angle alpha(t)
        at = (2*np.pi/3.154e7)*timearray
        ## Eccentricity. L-dependent, so needs to be altered for time-varied arm length case.
        e = L/(2*a*np.sqrt(3))
        ## Initialize arrays
        beta_n = np.array([0,0,0])
        x_n = np.array([0,0,0])
        y_n = np.array([0,0,0])
        z_n = np.array([0,0,0])
        ## Calculate inclination and positions for each satellite.
        for n in sats:
            beta_n[n-1] = (n-1) + (2/3)*np.pi + betaphase
            x_n[n-1] = a*np.cos(at) + a*e*(np.sin(at)*np.cos(at)*np.sin(beta_n[n-1]) - (1+np.sin(at)**2)*np.cos(beta_n[n-1]))
            y_n[n-1] = a*np.sin(at) + a*e*(np.sin(at)*np.cos(at)*np.sin(beta_n[n-1]) - (1+np.cos(at)**2)*np.sin(beta_n[n-1]))
            z_n[n-1] = -np.sqrt(3)*a*e*np.cos(at-beta_n[n-1])
        
        ## Construct position vectors r_n
        r1 = np.array([x_n[0],y_n[0],z_n[0]])
        r2 = np.array([x_n[1],y_n[1],z_n[1]])
        r3 = np.array([x_n[2],y_n[2],z_n[2]])
        
        
    def doppler_response(self, f0, theta, phi, r1, r2, r3, ti):
        
        '''
        Calculate Antenna pattern/ detector transfer function for a GW originating in the direction of (theta, phi) for the u doppler channel of an orbiting LISA with satellite position vectors r1, r2, r3 at a given time. Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and is input as an array
        

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        phi theta  :  float
            Sky position values. 
            
        r1, r2, r3  :  array
            Satellite position vectors.
        
        ti  :  float
            timearray index
    

        Returns
        ---------

        Rplus, Rcross   :   float
            Plus and cross antenna Patterns for the given sky direction
        '''

        ## Define cos/sin(theta)
        ct = np.cos(theta)
        st = np.sqrt(1-ct**2)
        ## Define x/y/z for each satellite at time given by timearray[ti]
        x1 = r1[0][ti]
        y1 = r1[1][ti]
        z1 = r1[2][ti]
        x2 = r2[0][ti]
        y2 = r2[1][ti]
        z2 = r2[2][ti]
        x3 = r3[0][ti]
        y3 = r3[1][ti]
        z3 = r3[2][ti]
        ## Define vector u at time timearray[ti]
        uvec = r2[:,ti] - r1[:,ti]
        ## Calculate arm length for the u arm
        Lu = np.dot(uvec,uvec)
        ## udir is just u-hat.omega, where u-hat is the u unit vector and omega is the unit vector in the sky direction of the GW signal
        udir = ((x2-x1)/Lu)*np.cos(phi)*st + ((y2-y1)/Lu)*np.sin(phi)*st + ((z2-z1)/Lu)*ct

        # Initlize arrays for the detector reponse
        Rplus, Rcross = np.zeros(f0.size), np.zeros(f0.size)

        # Calculate the detector response for each frequency

        # Calculate GW transfer function for the michelson channels
        gammaU = 1/2 * (np.sinc(f0*(1-udir))*np.exp(-1j*f0*(3+udir)) + \
                            np.sinc(f0*(1+udir))*np.exp(-1j*f0*(1+udir)))


        ## Michelson Channel Antenna patterns for + pol: Fplus_u = 1/2(u x u)Gamma(udir, f):eplus

        Rplus = 1/2*((((x2-x1)/Lu)*np.sin(phi)-((y2-y1)/Lu)*np.cos(phi))**2 - \
                     (((x2-x1)/Lu)*np.cos(phi)*ct+((y2-y1)/Lu)*np.sin(phi)*ct- \
                      ((z2-z1)/Lu)*st)**2)*gammaU
        
        ## Michelson Channel Antenna patterns for x pol
        ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross

        Rcross = ((((x2-x1)/Lu)*np.sin(phi)-((y2-y1)/Lu)*np.cos(phi)) * \
                  (((x2-x1)/Lu)*np.cos(phi)*ct+((y2-y1)/Lu)*np.sin(phi)*ct- \
                   ((z2-z1)/Lu)*st))*gammaU
        
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

    def tdi_isgwb_xyz_response(self, f0): 

        '''
        Calcualte the Antenna pattern/ detector transfer function functions to an isotropic SGWB using X, Y and Z TDI channels. Note that since this is the response to an isotropic background, the response function is integrated over sky direction and averaged over polarozation. The angular integral is a linear and rectangular in the cos(theta) and phi space.  Note that f0 is (pi*L*f)/c and is input as an array

        

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
            gammaU    =    1/2 * (np.sinc((f0[ii])*(1 - udir))*np.exp(-1j*f0[ii]*(3+udir)) + \
                             np.sinc((f0[ii])*(1 + udir))*np.exp(-1j*f0[ii]*(1+udir)))

            gammaV    =    1/2 * (np.sinc((f0[ii])*(1 - vdir))*np.exp(-1j*f0[ii]*(3+vdir)) + \
                             np.sinc((f0[ii])*(1 + vdir))*np.exp(-1j*f0[ii]*(1+vdir)))

            gammaW    =    1/2 * (np.sinc((f0[ii])*(1 - wdir))*np.exp(-1j*f0[ii]*(3+wdir)) + \
                             np.sinc((f0[ii])*(1 + wdir))*np.exp(-1j*f0[ii]*(1+wdir)))

            ## Michelson Channel Antenna patterns for + pol
            ##  Fplus_u = 1/2(u x u)Gamma(udir, f):eplus

            Fplus_u   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2) )# + \
                        #    0.5*((np.cos(phi))**2 - ct**2))*gammaU

            Fplus_v   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 + np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2) )# + \
                        # 0.5*((np.cos(phi))**2 - ct**2))*gammaV

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
            FXplus = 2*np.sin(2*f0[ii])*Fplus1
            FYplus = 2*np.sin(2*f0[ii])*Fplus2
            FZplus = 2*np.sin(2*f0[ii])*Fplus2

            FXcross = 2*np.sin(2*f0[ii])*Fcross1
            FYcross = 2*np.sin(2*f0[ii])*Fcross2
            FZcross = 2*np.sin(2*f0[ii])*Fcross3

            ## Detector response for the TDI Channels, summed over polarization
            ## and integrated over sky direction
            R1[ii] = dct*dphi/(4*np.pi)*np.sum((np.absolute(FXplus))**2 + (np.absolute(FXcross))**2)
            R2[ii] = dct*dphi/(4*np.pi)*np.sum((np.absolute(FYplus))**2 + (np.absolute(FYcross))**2)
            R3[ii] = dct*dphi/(4*np.pi)*np.sum((np.absolute(FZplus))**2 + (np.absolute(FZcross))**2)



        return R1, R2, R3




    def tdi_isgwb_response(self, f0): 

        '''
        Calcualte the Antenna pattern/ detector transfer function functions to an isotropic SGWB using A, E and T TDI channels. Note that since this is the response to an isotropic background, the response function is integrated over sky direction and averaged over polarozation. The angular integral is a linear and rectangular in the cos(theta) and phi space.  Note that f0 is (pi*L*f)/c and is input as an array

        

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
            gammaU    =    1/2 * (np.sinc((f0[ii])*(1 - udir))*np.exp(-1j*f0[ii]*(3+udir)) + \
                             np.sinc((f0[ii])*(1 + udir))*np.exp(-1j*f0[ii]*(1+udir)))

            gammaV    =    1/2 * (np.sinc((f0[ii])*(1 - vdir))*np.exp(-1j*f0[ii]*(3+vdir)) + \
                             np.sinc((f0[ii])*(1 + vdir))*np.exp(-1j*f0[ii]*(1+vdir)))

            gammaW    =    1/2 * (np.sinc((f0[ii])*(1 - wdir))*np.exp(-1j*f0[ii]*(3+wdir)) + \
                             np.sinc((f0[ii])*(1 + wdir))*np.exp(-1j*f0[ii]*(1+wdir)))

            ## Michelson Channel Antenna patterns for + pol
            ##  Fplus_u = 1/2(u x u)Gamma(udir, f):eplus

            Fplus_u   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2)) #+ \
                        #    0.5*((np.cos(phi))**2 - ct**2))*gammaU

            Fplus_v   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 + np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2)) #+ \
                        #     0.5*((np.cos(phi))**2 - ct**2))*gammaV

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

    def tdi_aniso_sph_sgwb_response(self, f0): 

        '''
        Calculate the Antenna pattern/ detector transfer function functions to acSGWB using A, E and T TDI channels, and using a spherical harmonic decomposition. Note that the response function is integrated over sky direction with the appropriate legandre polynomial, and averaged over polarozation. Finally note that the spherical harmonic coeffcients correspond to strain sky distribution, while the legandre polynomials describe the power sky. The angular integral is a linear and rectangular in the cos(theta) and phi space.  Note that f0 is (pi*L*f)/c and is input as an array

        

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

    

        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarization. The arrays are 2-d, one direction corresponds to frequency and the other to the l coeffcient. 
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
            
            R1[ii, :] = dct*dphi/(4*np.pi)*np.sum(np.tensordot((np.absolute(FAplus))**2 + \
                    (np.absolute(FAcross))**2, plms, axes=1), axis=(0, 1))
            R2[ii, :] = dct*dphi/(4*np.pi)*np.sum(np.tensordot((np.absolute(FEplus))**2 + \
                    (np.absolute(FEcross))**2, plms, axes=1), axis=(0, 1))
            R3[ii, :] = dct*dphi/(4*np.pi)*np.sum(np.tensordot((np.absolute(FTplus))**2 + \
                    (np.absolute(FTcross))**2, plms, axes=1), axis=(0,1))   



        return R1, R2, R3


    def fundamental_noise_spectrum(self, freqs, Np=4e-41, Na=1.44e-48):

        '''
        Creates a frequency array of fundamentla noise estimates for lisa. Currently we consisder only contain only position and acceleration noise sources. The default values are specifications pulled from 2017 Lisa proposal noise estimations.

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


