import numpy as np
from scipy.special import lpmn, sph_harm
import types
import healpy as hp
#from src.orbitinglisa import orbitinglisa
from src.sph_geometry import sph_geometry

class geometry(sph_geometry):

    '''
    Module containing geometry methods. The methods here include calculation of antenna patters for a single doppler channel, for the three michelson channels or for the AET TDI channels and calculation of noise power spectra for various channel combinations.
    '''

    def __init__(self):

        if self.params['modeltype'] == 'sph_sgwb' or self.inj['injtype'] == 'sph_sgwb':
            sph_geometry.__init__(self)


    def lisa_orbits(self, tsegmid, tsegstart):

        '''
        Define LISA orbital positions at the midpoint of each time integration segment using analytic MLDC orbits.

        Parameters
        -----------

        tsegmid  :  array
            A numpy array of the tsegmid for each time integration segment.

        Returns
        -----------
        rs1, rs2, rs3  :  array
            Arrays of satellite positions for each segment midpoint in timearray. e.g. rs1[1] is [x1,y1,z1] at t=midpoint[1]=timearray[1]+(segment length)/2.
        '''
        ## Branch orbiting and stationary cases; compute satellite position in stationary case based off of first time entry in data.
        if self.params['lisa_config'] == 'stationary':
            times = tsegmid.fill(tsegstart[0])
        elif self.params['lisa_config'] == 'orbiting':
            times = tsegmid
        else:
            raise ValueError('Unknown LISA configuration selected')


        ## Semimajor axis in m
        a = 1.496e11


        ## Alpha and beta phases allow for changing of initial satellite orbital phases; default initial conditions are alphaphase=betaphase=0.
        betaphase = 0
        alphaphase = 0

        ## Orbital angle alpha(t)
        at = (2*np.pi/31557600)*times + alphaphase

        ## Eccentricity. L-dependent, so needs to be altered for time-varied arm length case.
        e = self.armlength/(2*a*np.sqrt(3))

        ## Initialize arrays
        beta_n = (2/3)*np.pi*np.array([0,1,2])+betaphase

        ## meshgrid arrays
        Beta_n, Alpha_t = np.meshgrid(beta_n, at)

        ## Calculate inclination and positions for each satellite.
        x_n = a*np.cos(Alpha_t) + a*e*(np.sin(Alpha_t)*np.cos(Alpha_t)*np.sin(Beta_n) - (1+(np.sin(Alpha_t))**2)*np.cos(Beta_n))
        y_n = a*np.sin(Alpha_t) + a*e*(np.sin(Alpha_t)*np.cos(Alpha_t)*np.cos(Beta_n) - (1+(np.cos(Alpha_t))**2)*np.sin(Beta_n))
        z_n = -np.sqrt(3)*a*e*np.cos(Alpha_t - Beta_n)


        ## Construct position vectors r_n
        rs1 = np.array([x_n[:, 0],y_n[:, 0],z_n[:, 0]])
        rs2 = np.array([x_n[:, 1],y_n[:, 1],z_n[:, 1]])
        rs3 = np.array([x_n[:, 2],y_n[:, 2],z_n[:, 2]])

        return rs1, rs2, rs3

#    def doppler_response(self, f0, theta, phi):
#
#        '''
#        Calculate Antenna pattern/ detector transfer function for a GW originating in the direction of (theta, phi) for the doppler channel of a stationary LISA. Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and is input as an array
#
#
#        Parameters
#        -----------
#
#        f0   : float
#            A numpy array of scaled frequencies (see above for def)
#
#        phi theta  : float
#            Sky position values.
#
#
#        Returns
#        ---------
#
#        Rplus, Rcorss   :   float
#            Plus and cross antenna Patterns for the given sky direction
#        '''
#
#
#        ct = np.cos(theta)
#
#        ## udir is just u.r, where r is the directional vector
#        udir = np.sqrt(1-ct**2) * np.sin(phi + np.pi/6)
#
#        # Initlize arrays for the detector reponse
#        Rplus, Rcross = np.zeros(f0.size), np.zeros(f0.size)
#
#        # Calculate the detector response for each frequency
#
#        # Calculate GW transfer function for the michelson channels
#        gammaU    =    1/2 * (np.sinc(f0*(1 - udir)/np.pi)*np.exp(-1j*f0*(3+udir)) + \
#                            np.sinc(f0*(1 + udir)/np.pi)*np.exp(-1j*f0*(1+udir)))
#
#
#        ## Michelson Channel Antenna patterns for + pol: Fplus_u = 1/2(u x u)Gamma(udir, f):eplus
#
#        Rplus   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2) + \
#                            0.5*((np.cos(phi))**2 - ct**2))*gammaU
#
#        ## Michelson Channel Antenna patterns for x pol
#        ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross
#
#        Rcross = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi + np.pi/3))*gammaU
#
#        return Rplus, Rcross

    def doppler_response(self, f0, theta, phi, tsegmid, tsegstart):

        '''
        Calculate antenna pattern/ detector transfer functions for a GW originating in the direction of (theta, phi) for the u doppler channel of an orbiting LISA with satellite position vectors rs1, rs2, rs3. Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and is input as an array.


        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        phi theta  :  float
            Sky position values.

        tsegmid  :  array
            A numpy array of the midpoints for each time integration segment.

        rs1, rs2, rs3  :  array
            Satellite position vectors.


        Returns
        ---------

        Rplus, Rcross   :   float
            Plus and cross antenna Patterns for the given sky direction for each time in midpoints.
        '''
        print('Calculating detector response functions...')

        self.rs1, self.rs2, self.rs3 = self.lisa_orbits(tsegmid, tsegstart)

        ## Indices of midpoints array
        timeindices = np.arange(len(tsegmid))

        ## Define cos/sin(theta)
        ct = np.cos(theta)
        st = np.sqrt(1-ct**2)

        ## Initlize arrays for the detector reponse
        Rplus, Rcross = np.zeros((len(timeindices),f0.size), dtype=complex), np.zeros((len(timeindices),f0.size),dtype=complex)

        for ti in timeindices:
            ## Define x/y/z for each satellite at time given by timearray[ti]
            x1 = rs1[0][ti]
            y1 = rs1[1][ti]
            z1 = rs1[2][ti]
            x2 = rs2[0][ti]
            y2 = rs2[1][ti]
            z2 = rs2[2][ti]

            ## Add if calculating v, w:
            ## x3 = r3[0][ti]
            ## y3 = r3[1][ti]
            ## z3 = r3[2][ti]

            ## Define vector u at time tsegmid[ti]
            uvec = rs2[:,ti] - rs1[:,ti]
            ## Calculate arm length for the u arm
            Lu = np.sqrt(np.dot(uvec,uvec))
            ## udir is just u-hat.omega, where u-hat is the u unit vector and omega is the unit vector in the sky direction of the GW signal
            udir = ((x2-x1)/Lu)*np.cos(phi)*st + ((y2-y1)/Lu)*np.sin(phi)*st + ((z2-z1)/Lu)*ct

            ## Calculate 1/2(u x u):eplus
            Pcontract = 1/2*((((x2-x1)/Lu)*np.sin(phi)-((y2-y1)/Lu)*np.cos(phi))**2 - \
                             (((x2-x1)/Lu)*np.cos(phi)*ct+((y2-y1)/Lu)*np.sin(phi)*ct- \
                              ((z2-z1)/Lu)*st)**2)
             ## Calculate 1/2(u x u):ecross
            Ccontract = ((((x2-x1)/Lu)*np.sin(phi)-((y2-y1)/Lu)*np.cos(phi)) * \
                          (((x2-x1)/Lu)*np.cos(phi)*ct+((y2-y1)/Lu)*np.sin(phi)*ct- \
                           ((z2-z1)/Lu)*st))

            # Calculate the detector response for each frequency
            for ii in range(0, f0.size):
                # Calculate GW transfer function for the michelson channels
                gammaU = 1/2 * (np.sinc(f0[ii]*(1-udir)/np.pi)*np.exp(-1j*f0[ii]*(3+udir)) + \
                                    np.sinc(f0[ii]*(1+udir)/np.pi)*np.exp(-1j*f0[ii]*(1+udir)))


                ## Michelson Channel Antenna patterns for + pol: Rplus = 1/2(u x u)Gamma(udir, f):eplus

                Rplus[ti][ii] = Pcontract*gammaU

                ## Michelson Channel Antenna patterns for x pol: Rcross = 1/2(u x u)Gamma(udir, f):ecross

                Rcross[ti][ii] = Ccontract*gammaU

        return Rplus, Rcross


#    def michelson_response(self, f0, theta, phi):
#
#        '''
#        Calculate Antenna pattern/ detector transfer function for a GW originating in the direction of (theta, phi) for the three Michelson channels of a stationary LISA. Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and is input as an array
#
#
#        Parameters
#        -----------
#
#        f0   : float
#            A numpy array of scaled frequencies (see above for def)
#
#        phi theta  : float
#            Sky position values.
#
#
#        Returns
#        ---------
#
#        R1plus, R1cross, R2plus, R2cross, R3plus, R3cross   :   float
#            Plus and cross antenna Patterns for the given sky direction for the three channels
#        '''
#
#        ct = np.cos(theta)
#
#        ## udir is just u.r, where r is the directional vector
#        udir = np.sqrt(1-ct**2) * np.sin(phi + np.pi/6)
#        vdir = np.sqrt(1-ct**2) * np.sin(phi - np.pi/6)
#        wdir = vdir - udir
#
#        # Calculate GW transfer function for the michelson channels
#        gammaU    =    1/2 * (np.sinc(f0*(1 - udir)/np.pi)*np.exp(-1j*f0*(3+udir)) + \
#                         np.sinc((f0)*(1 + udir)/np.pi)*np.exp(-1j*f0*(1+udir)))
#
#        gammaV    =    1/2 * (np.sinc(f0*(1 - vdir)/np.pi)*np.exp(-1j*f0*(3+vdir)) + \
#                         np.sinc((f0)*(1 + vdir)/np.pi)*np.exp(-1j*f0*(1+vdir)))
#
#        gammaW    =    1/2 * (np.sinc(f0*(1 - wdir)/np.pi)*np.exp(-1j*f0*(3+wdir)) + \
#                         np.sinc((f0)*(1 + wdir)/np.pi)*np.exp(-1j*f0*(1+wdir)))
#
#        ## Michelson Channel Antenna patterns for + pol
#        ##  Fplus_u = 1/2(u x u)Gamma(udir, f):eplus
#        Fplus_u   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2) + \
#                        0.5*((np.cos(phi))**2 - ct**2))*gammaU
#
#        Fplus_v   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 + np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2)+ \
#                     0.5*((np.cos(phi))**2 - ct**2))*gammaV
#
#        Fplus_w   = 1/2*(1 - (1+ct**2)*(np.cos(phi))**2)*gammaW
#
#        ## Michelson Channel Antenna patterns for x pol
#        ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross
#
#        Fcross_u  = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi + np.pi/3))*gammaU
#        Fcross_v  = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi - np.pi/3))*gammaV
#        Fcross_w   = 1/2*ct*np.sin(2*phi)*gammaW
#
#        ## Compelte Michelson antenna patterns
#        ## Calculate Fplus
#        R1plus = (Fplus_u - Fplus_v)
#        R2plus = (Fplus_w - Fplus_u)
#        R3plus = (Fplus_v - Fplus_w)
#
#        ## Calculate Fcross
#        R1cross = (Fcross_u - Fcross_v)
#        R2cross = (Fcross_w - Fcross_u)
#        R3cross = (Fcross_v - Fcross_w)
#
#        return R1plus, R1cross, R2plus, R2cross, R3plus, R3cross

    def michelson_response(self, f0, theta, phi, tsegmid, tsegstart):

        '''
        Calculate Antenna pattern/ detector transfer function for a GW originating in the direction of (theta, phi) at a given time for the three Michelson channels of an orbiting LISA. Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and is input as an array


        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        phi theta  : float
            Sky position values.

        rs1, rs2, rs3  :  arrays
            Satellite position vectors.

        tsegmid  :  array
            A numpy array of the midpoints for each time integration segment.


        Returns
        ---------

        R1plus, R1cross, R2plus, R2cross, R3plus, R3cross   :   arrays
            Plus and cross antenna Patterns for the given sky direction for the three channels for each time in midpoints.
        '''
        print('Calculating detector response functions...')

        self.rs1, self.rs2, self.rs3 = self.lisa_orbits(tsegmid, tsegstart)

        ## Indices of midpoints array
        timeindices = np.arange(len(tsegmid))

        ## Define cos/sin(theta)
        ct = np.cos(theta)
        st = np.sqrt(1-ct**2)

        for ti in timeindices:
            ## Define x/y/z for each satellite at time given by tsegmid[ti]
            x1 = rs1[0][ti]
            y1 = rs1[1][ti]
            z1 = rs1[2][ti]
            x2 = rs2[0][ti]
            y2 = rs2[1][ti]
            z2 = rs2[2][ti]
            x3 = rs3[0][ti]
            y3 = rs3[1][ti]
            z3 = rs3[2][ti]

            ## Define vector u at time timearray[ti]
            uvec = rs2[:,ti] - rs1[:,ti]
            vvec = rs3[:,ti] - rs1[:,ti]
            wvec = rs3[:,ti] - rs2[:,ti]

            ## Calculate arm lengths
            Lu = np.sqrt(np.dot(uvec,uvec))
            Lv = np.sqrt(np.dot(vvec,vvec))
            Lw = np.sqrt(np.dot(wvec,wvec))

            ## udir is just u-hat.omega, where u-hat is the u unit vector and omega is the unit vector in the sky direction of the GW signal
            udir = ((x2-x1)/Lu)*np.cos(phi)*st + ((y2-y1)/Lu)*np.sin(phi)*st + ((z2-z1)/Lu)*ct
            vdir = ((x3-x1)/Lv)*np.cos(phi)*st + ((y3-y1)/Lv)*np.sin(phi)*st + ((z3-z1)/Lv)*ct
            wdir = ((x3-x2)/Lw)*np.cos(phi)*st + ((y3-y2)/Lw)*np.sin(phi)*st + ((z3-z2)/Lw)*ct

            ## Calculate 1/2(u x u):eplus
            Pcontract_u = 1/2*((((x2-x1)/Lu)*np.sin(phi)-((y2-y1)/Lu)*np.cos(phi))**2 - \
                             (((x2-x1)/Lu)*np.cos(phi)*ct+((y2-y1)/Lu)*np.sin(phi)*ct-((z2-z1)/Lu)*st)**2)
            Pcontract_v = 1/2*((((x3-x1)/Lv)*np.sin(phi)-((y3-y1)/Lv)*np.cos(phi))**2 - \
                             (((x3-x1)/Lv)*np.cos(phi)*ct+((y3-y1)/Lv)*np.sin(phi)*ct-((z3-z1)/Lv)*st)**2)
            Pcontract_w = 1/2*((((x3-x2)/Lw)*np.sin(phi)-((y3-y2)/Lw)*np.cos(phi))**2 - \
                             (((x3-x2)/Lw)*np.cos(phi)*ct+((y3-y2)/Lw)*np.sin(phi)*ct-((z3-z2)/Lw)*st)**2)

            ## Calculate 1/2(u x u):ecross
            Ccontract_u = (((x2-x1)/Lu)*np.sin(phi)-((y2-y1)/Lu)*np.cos(phi)) * \
                            (((x2-x1)/Lu)*np.cos(phi)*ct+((y2-y1)/Lu)*np.sin(phi)*ct-((z2-z1)/Lu)*st)

            Ccontract_v = (((x3-x1)/Lv)*np.sin(phi)-((y3-y1)/Lv)*np.cos(phi)) * \
                            (((x3-x1)/Lv)*np.cos(phi)*ct+((y3-y1)/Lv)*np.sin(phi)*ct-((z3-z1)/Lv)*st)

            Ccontract_w = (((x3-x2)/Lw)*np.sin(phi)-((x3-x2)/Lw)*np.cos(phi)) * \
                            (((x3-x2)/Lw)*np.cos(phi)*ct+((y3-y2)/Lw)*np.sin(phi)*ct-((z3-z2)/Lw)*st)


            ## Calculate the detector response for each frequency
            for ii in range(0, f0.size):

                ## Calculate GW transfer function for the michelson channels
                gammaU_p    =    1/2 * (np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(3 + udir)) + \
                                        np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(1 + udir)))
                gammaU_m    =    1/2 * (np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(3 - udir)) + \
                                        np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(1 - udir)))

                gammaV_p    =    1/2 * (np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(3 + vdir)) + \
                                        np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(1+vdir)))
                gammaV_m    =    1/2 * (np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(3 - vdir)) + \
                                        np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(1 - vdir)))

                gammaW_p    =    1/2 * (np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(3 + wdir)) + \
                                        np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(1 + wdir)))
                gammaW_m    =    1/2 * (np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(3 - wdir)) + \
                                        np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(1 - wdir)))
                ## Michelson Channel Antenna patterns for + pol
                ## Fplus_u = 1/2(u x u)Gamma(udir, f):eplus

                Fplus_u_p   = Pcontract_u*gammaU_p
                Fplus_u_m   = Pcontract_u*gammaU_m
                Fplus_v_p   = Pcontract_v*gammaV_p
                Fplus_v_m   = Pcontract_v*gammaV_m
                Fplus_w_p   = Pcontract_w*gammaW_p
                Fplus_w_m   = Pcontract_w*gammaW_m

                ## Michelson Channel Antenna patterns for x pol
                ## Fcross_u = 1/2(u x u)Gamma(udir, f):ecross
                Fcross_u_p  = Ccontract_u*gammaU_p
                Fcross_u_m  = Ccontract_u*gammaU_m
                Fcross_v_p  = Ccontract_v*gammaV_p
                Fcross_v_m  = Ccontract_v*gammaV_m
                Fcross_w_p  = Ccontract_w*gammaW_p
                Fcross_w_m  = Ccontract_w*gammaW_m


                ## First Michelson antenna patterns
                ## Calculate Fplus
                R1plus = (Fplus_u_p - Fplus_v_p)
                R2plus = (Fplus_w_p - Fplus_u_m)
                R3plus = (Fplus_v_m - Fplus_w_m)

                ## Calculate Fcross
                R1cross = (Fcross_u_p - Fcross_v_p)
                R2cross = (Fcross_w_p - Fcross_u_m)
                R3cross = (Fcross_v_m - Fcross_w_m)


        return R1plus, R1cross, R2plus, R2cross, R3plus, R3cross

#    def aet_response(self, f0, theta, phi):
#
#
#
#        '''
#        Calculate Antenna pattern/ detector transfer function for a GW originating in the direction of (theta, phi) for the A, E and T TDI channels of a stationary LISA. Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and is input as an array
#
#
#        Parameters
#        -----------
#
#        f0   : float
#            A numpy array of scaled frequencies (see above for def)
#
#        phi theta  : float
#            Sky position values.
#
#
#        Returns
#        ---------
#
#        RAplus, RAcross, REplus, REcross, RTplus, RTcross   :   float
#            Plus and cross antenna Patterns for the given sky direction for the three channels
#        '''
#
#
#        R1plus, R1cross, R2plus, R2cross, R3plus, R3cross  = self.michelson_response(f0, theta, phi)
#
#
#        ## Calculate antenna patterns for the A, E and T channels
#        RAplus = (2/3)*np.sin(2*f0)*(2*R1plus - R2plus - R3plus)
#        REplus = (2/np.sqrt(3))*np.sin(2*f0)*(R3plus - R2plus)
#        RTplus = (1/3)*np.sin(2*f0)*(R1plus + R3plus + R2plus)
#
#        RAcross = (2/3)*np.sin(2*f0)*(2*R1cross - R2cross - R3cross)
#        REcross = (2/np.sqrt(3))*np.sin(2*f0)*(R3cross - R2cross)
#        RTcross = (1/3)*np.sin(2*f0)*(R1cross + R3cross + R2cross)
#
#        return RAplus, RAcross, REplus, REcross, RTplus, RTcross

        def aet_response(self, f0, theta, phi, tsegmid, tsegstart):



        '''
        Calculate Antenna pattern/ detector transfer functions for a GW originating in the direction of (theta, phi) for the A, E and T TDI channels of an orbiting LISA. Return the detector responses for + and x polarization. Note that f0 is (pi*L*f)/c and is input as an array


        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        phi theta  : float
            Sky position values.

        tsegmid  :  array
            A numpy array of the midpoints for each time integration segment.

        rs1, rs2, rs3  :  array
            Satellite position vectors.


        Returns
        ---------

        RAplus, RAcross, REplus, REcross, RTplus, RTcross   :   arrays
            Plus and cross antenna Patterns for the given sky direction for the three channels for each time in midpoints.
        '''

        self.rs1, self.rs2, self.rs3 = self.lisa_orbits(tsegmid, tsegstart)

        R1plus, R1cross, R2plus, R2cross, R3plus, R3cross  = self.orbiting_michelson_response(f0, theta, phi, tsegmid, tsegstart)


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



        npix = hp.nside2npix(self.params['nside'])

        # Array of pixel indices
        pix_idx  = np.arange(npix)

        #Angular coordinates of pixel indcides
        theta, phi = hp.pix2ang(self.params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)

        # Area of each pixel in sq.radians
        dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])

        ## Create directional vectors
        udir = np.sqrt(1-ctheta**2) * np.sin(phi + np.pi/6)
        vdir = np.sqrt(1-ctheta**2) * np.sin(phi - np.pi/6)
        wdir = vdir - udir

        # Initlize arrays for the detector reponse
        R1 = np.zeros(f0.size, dtype='complex')
        R2 = np.zeros(f0.size, dtype='complex')
        R3 = np.zeros(f0.size, dtype='complex')
        R12 = np.zeros(f0.size, dtype='complex')
        R13 = np.zeros(f0.size, dtype='complex')
        R23 = np.zeros(f0.size, dtype='complex')

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
            Fplus1 = 0.5*(Fplus_u*gammaU_plus - Fplus_v*gammaV_plus)*np.exp(-1j*f0[ii]*(udir + vdir)/np.sqrt(3))
            Fplus2 = 0.5*(Fplus_w*gammaW_plus - Fplus_u*gammaU_minus)*np.exp(-1j*f0[ii]*(-udir + vdir)/np.sqrt(3))
            Fplus3 = 0.5*(Fplus_v*gammaV_minus - Fplus_w*gammaW_minus)*np.exp(1j*f0[ii]*(vdir + wdir)/np.sqrt(3))

            ## Calculate Fcross
            Fcross1 = 0.5*(Fcross_u*gammaU_plus  - Fcross_v*gammaV_plus)*np.exp(-1j*f0[ii]*(udir + vdir)/np.sqrt(3))
            Fcross2 = 0.5*(Fcross_w*gammaW_plus  - Fcross_u*gammaU_minus)*np.exp(-1j*f0[ii]*(-udir + vdir)/np.sqrt(3))
            Fcross3 = 0.5*(Fcross_v*gammaV_minus - Fcross_w*gammaW_minus)*np.exp(1j*f0[ii]*(vdir + wdir)/np.sqrt(3))

            ## Detector response summed over polarization and integrated over sky direction
            ## The travel time phases for the which are relevent for the cross-channel are
            ## accounted for in the Fplus and Fcross expressions above.

            R1[ii]  = dOmega/(8*np.pi)*np.sum( (np.absolute(Fplus1))**2 + (np.absolute(Fcross1))**2 )
            R2[ii]  = dOmega/(8*np.pi)*np.sum( (np.absolute(Fplus2))**2 + (np.absolute(Fcross2))**2 )
            R3[ii]  = dOmega/(8*np.pi)*np.sum( (np.absolute(Fplus3))**2 + (np.absolute(Fcross3))**2 )
            R12[ii] = dOmega/(8*np.pi)*np.sum( np.conj(Fplus1)*Fplus2 + np.conj(Fcross1)*Fcross2)
            R13[ii] = dOmega/(8*np.pi)*np.sum( np.conj(Fplus1)*Fplus3 + np.conj(Fcross1)*Fcross3)
            R23[ii] = dOmega/(8*np.pi)*np.sum( np.conj(Fplus2)*Fplus3 + np.conj(Fcross2)*Fcross3)

        response_mat = np.array([ [R1, R12, R13] , [np.conj(R12), R2, R23], [np.conj(R13), np.conj(R23), R3] ])

        return response_mat

    def isgwb_mich_response_combined(self, f0):

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



        npix = hp.nside2npix(self.params['nside'])

        # Array of pixel indices
        pix_idx  = np.arange(npix)

        #Angular coordinates of pixel indcides
        theta, phi = hp.pix2ang(self.params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)

        # Area of each pixel in sq.radians
        dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])

        ## Create directional vectors
        udir = np.sqrt(1-ctheta**2) * np.sin(phi + np.pi/6)
        vdir = np.sqrt(1-ctheta**2) * np.sin(phi - np.pi/6)
        wdir = vdir - udir

        # Initlize arrays for the detector reponse
        R1 = np.zeros(f0.size, dtype='complex')
        R2 = np.zeros(f0.size, dtype='complex')
        R3 = np.zeros(f0.size, dtype='complex')
        R12 = np.zeros(f0.size, dtype='complex')
        R13 = np.zeros(f0.size, dtype='complex')
        R23 = np.zeros(f0.size, dtype='complex')

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
            Fplus1 = 0.5*(Fplus_u*gammaU_plus - Fplus_v*gammaV_plus)*np.exp(-1j*f0[ii]*(udir + vdir)/np.sqrt(3))
            Fplus2 = 0.5*(Fplus_w*gammaW_plus - Fplus_u*gammaU_minus)*np.exp(-1j*f0[ii]*(-udir + vdir)/np.sqrt(3))
            Fplus3 = 0.5*(Fplus_v*gammaV_minus - Fplus_w*gammaW_minus)*np.exp(1j*f0[ii]*(vdir + wdir)/np.sqrt(3))

            ## Calculate Fcross
            Fcross1 = 0.5*(Fcross_u*gammaU_plus  - Fcross_v*gammaV_plus)*np.exp(-1j*f0[ii]*(udir + vdir)/np.sqrt(3))
            Fcross2 = 0.5*(Fcross_w*gammaW_plus  - Fcross_u*gammaU_minus)*np.exp(-1j*f0[ii]*(-udir + vdir)/np.sqrt(3))
            Fcross3 = 0.5*(Fcross_v*gammaV_minus - Fcross_w*gammaW_minus)*np.exp(1j*f0[ii]*(vdir + wdir)/np.sqrt(3))

            ## Detector response summed over polarization and integrated over sky direction
            ## The travel time phases for the which are relevent for the cross-channel are
            ## accounted for in the Fplus and Fcross expressions above.

            R1[ii]  = dOmega/(8*np.pi)*np.sum( (np.absolute(Fplus1))**2 + (np.absolute(Fcross1))**2 )
            R2[ii]  = dOmega/(8*np.pi)*np.sum( (np.absolute(Fplus2))**2 + (np.absolute(Fcross2))**2 )
            R3[ii]  = dOmega/(8*np.pi)*np.sum( (np.absolute(Fplus3))**2 + (np.absolute(Fcross3))**2 )
            R12[ii] = dOmega/(8*np.pi)*np.sum( np.conj(Fplus1)*Fplus2 + np.conj(Fcross1)*Fcross2)
            R13[ii] = dOmega/(8*np.pi)*np.sum( np.conj(Fplus1)*Fplus3 + np.conj(Fcross1)*Fcross3)
            R23[ii] = dOmega/(8*np.pi)*np.sum( np.conj(Fplus2)*Fplus3 + np.conj(Fcross2)*Fcross3)

        response_mat = np.array([ [R1, R12, R13] , [np.conj(R12), R2, R23], [np.conj(R13), np.conj(R23), R3] ])

        return response_mat


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

        mich_response_mat = self.isgwb_mich_response(f0)
        xyz_response_mat = 4 * mich_response_mat * (np.sin(2*f0[None, None, :]))**2

        return xyz_response_mat


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


        Return      s
        ---------
        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarization.
        '''

        xyz_response_mat = self.isgwb_xyz_response(f0)

        ## Upnack xyz matrix to make assembling the aet matrix easier
        RXX, RYY, RZZ = xyz_response_mat[0, 0], xyz_response_mat[1, 1], xyz_response_mat[2, 2]
        RXY, RXZ, RYZ = xyz_response_mat[0, 1], xyz_response_mat[0, 2], xyz_response_mat[1, 2]


        ## construct AET matrix elements
        RAA = (1/9) * (4*RXX + RYY + RZZ - 2*RXY - 2*np.conj(RXY) - 2*RXZ - 2*np.conj(RXZ) + \
                        RYZ  + np.conj(RYZ))

        REE = (1/3) * (RZZ + RYY - RYZ - np.conj(RYZ))

        RTT = (1/9) * (RXX + RYY + RZZ + RXY + np.conj(RXY) + RXZ + np.conj(RXZ) + RYZ + np.conj(RYZ))

        RAE = (1/(3*np.sqrt(3))) * (RYY - RZZ - RYZ + np.conj(RYZ) + 2*RXZ - 2*RXY)

        RAT = (1/9) * (2*RXX - RYY - RZZ + 2*RXY - np.conj(RXY) + 2*RXZ - np.conj(RXZ) - RYZ - np.conj(RYZ))

        RET = (1/(3*np.sqrt(3))) * (RZZ - RYY - RYZ + np.conj(RYZ) + np.conj(RXZ) - np.conj(RXY))

        aet_response_mat = np.array([ [RAA, RAE, RAT] , \
                                    [np.conj(RAE), REE, RET], \
                                    [np.conj(RAT), np.conj(RET), RTT] ])

        return aet_response_mat



    ## ------------------------ The methods above help calculate pattern functions useful for recovery --------------------------------
    ## ---------------------------------- The methods below help in making simualted data ---------------------------------------------

    ## ------------------------------------------- Anisotropic injection methods from below ----------------------------------------------------
    def R_sky_response(self, f0):

        '''
        Calculate the detector transfer function functions to an non-polarized SGWB non-polarized for basic michelson
        channels in a healpix grid. The output are six matrices which have the shape of npix x nfreqs. Note also that
        f0 is (pi*L*f)/c and is input as an array. The response function is given for the strain of the signal rather
        than the power



        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        Returns
        ---------

        R1_plus, R2_plus and R3_plus, R1_cross, R2_corss, R3_cross  :  complex
            Antenna Patterns for the healpix array for the three channels
        '''

        npix = hp.nside2npix(self.params['nside'])

        # Array of pixel indices
        pix_idx  = np.arange(npix)

        #Angular coordinates of pixel indcides
        theta, phi = hp.pix2ang(self.params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)

        # Area of each pixel in sq.radians
        dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])

        ## Create directional vectors
        udir = np.sqrt(1-ctheta**2) * np.sin(phi + np.pi/6)
        vdir = np.sqrt(1-ctheta**2) * np.sin(phi - np.pi/6)
        wdir = vdir - udir

        # Initlize arrays for the detector reponse
        R1_plus = np.zeros((f0.size, npix), dtype='complex')
        R2_plus = np.zeros((f0.size, npix), dtype='complex')
        R3_plus = np.zeros((f0.size, npix), dtype='complex')
        R1_cross = np.zeros((f0.size, npix), dtype='complex')
        R2_cross = np.zeros((f0.size, npix), dtype='complex')
        R3_cross = np.zeros((f0.size, npix), dtype='complex')

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
            Fplus1 = 0.5*(Fplus_u*gammaU_plus - Fplus_v*gammaV_plus)*np.exp(-1j*f0[ii]*(udir + vdir)/np.sqrt(3))
            Fplus2 = 0.5*(Fplus_w*gammaW_plus - Fplus_u*gammaU_minus)*np.exp(-1j*f0[ii]*(-udir + vdir)/np.sqrt(3))
            Fplus3 = 0.5*(Fplus_v*gammaV_minus - Fplus_w*gammaW_minus)*np.exp(1j*f0[ii]*(vdir + wdir)/np.sqrt(3))

            ## Calculate Fcross
            Fcross1 = 0.5*(Fcross_u*gammaU_plus  - Fcross_v*gammaV_plus)*np.exp(-1j*f0[ii]*(udir + vdir)/np.sqrt(3))
            Fcross2 = 0.5*(Fcross_w*gammaW_plus  - Fcross_u*gammaU_minus)*np.exp(-1j*f0[ii]*(-udir + vdir)/np.sqrt(3))
            Fcross3 = 0.5*(Fcross_v*gammaV_minus - Fcross_w*gammaW_minus)*np.exp(1j*f0[ii]*(vdir + wdir)/np.sqrt(3))

            ## Detector response summed over polarization and integrated over sky direction
            R1_plus[ii,:], R1_cross[ii, :] = np.sqrt(0.5/npix)*Fplus1, np.sqrt(0.5/npix)*Fcross1
            R2_plus[ii,:], R2_cross[ii, :] = np.sqrt(0.5/npix)*Fplus2, np.sqrt(0.5/npix)*Fcross2
            R3_plus[ii,:], R3_cross[ii, :] = np.sqrt(0.5/npix)*Fplus3, np.sqrt(0.5/npix)*Fcross3


        self.R1_plus, self.R1_cross, self.R2_plus, self.R2_cross, self.R3_plus, self.R3_cross = R1_plus, R1_cross,R2_plus, R2_cross,R3_plus, R3_cross


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

        R1, R2 and R3   :   complex
            Antenna Patterns for the three channels, integrated over sky direction and averaged over polarization.
        '''

        ## check if the directional pattern functions exist
        try:
            self.R1_plus
        except:
            self.R_sky_response(f0)

        npix = hp.nside2npix(self.params['nside'])

        # Assign random complex amplitudes for each pixel
        rand_plus  = np.random.standard_normal(size=(f0.size, npix)) + 1j*np.random.standard_normal(size=(f0.size, npix))
        rand_cross = np.random.standard_normal(size=(f0.size, npix)) + 1j*np.random.standard_normal(size=(f0.size, npix))

        # Initlize arrays for the detector reponse
        R1 = np.zeros((f0.size, 2), dtype='complex')
        R2 = np.zeros((f0.size, 2), dtype='complex')
        R3 = np.zeros((f0.size, 2), dtype='complex')


        ## dot with the phases and calculate the pattern functions
        R1[:, 0], R1[:, 1] = np.sum(self.R1_plus*rand_plus, axis=1), np.sum(self.R1_cross*rand_cross, axis=1)
        R2[:, 0], R2[:, 1] = np.sum(self.R2_plus*rand_plus, axis=1), np.sum(self.R2_cross*rand_cross, axis=1)
        R3[:, 0], R3[:, 1] = np.sum(self.R3_plus*rand_plus, axis=1), np.sum(self.R3_cross*rand_cross, axis=1)




        return R1, R2, R3

    def isgwb_xyz_strain_response(self, f0):

        '''
        Calculate the detector transfer function functions to an isotropic SGWB non-polarized using XYZ TDI
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

        R1, R2 and R3   :   complex
            Antenna Patterns for the three channels, integrated over sky direction and averaged over polarization.
        '''

        ## check if the directional pattern functions exist
        try:
            self.R1_plus
        except:
            self.R_sky_response(f0)

        npix = hp.nside2npix(self.params['nside'])

        # Assign random complex amplitudes for each pixel
        rand_plus  = np.random.standard_normal(size=(f0.size, npix)) + 1j*np.random.standard_normal(size=(f0.size, npix))
        rand_cross = np.random.standard_normal(size=(f0.size, npix)) + 1j*np.random.standard_normal(size=(f0.size, npix))

        ## Initlize arrays for the detector reponse
        ## These are XYZ TDI Channels!
        R1 = np.zeros((f0.size, 2), dtype='complex')
        R2 = np.zeros((f0.size, 2), dtype='complex')
        R3 = np.zeros((f0.size, 2), dtype='complex')


        ## dot with the phases and calculate the pattern functions
        ## The factor of sin(2*f0) comes because there are XYZ channels
        ## rather than being michelson channels
        R1[:, 0], R1[:, 1] = 2*np.sin(2*f0)*np.sum(self.R1_plus*rand_plus, axis=1), 2*np.sin(2*f0)*np.sum(self.R1_cross*rand_cross, axis=1)
        R2[:, 0], R2[:, 1] = 2*np.sin(2*f0)*np.sum(self.R2_plus*rand_plus, axis=1), 2*np.sin(2*f0)*np.sum(self.R2_cross*rand_cross, axis=1)
        R3[:, 0], R3[:, 1] = 2*np.sin(2*f0)*np.sum(self.R3_plus*rand_plus, axis=1), 2*np.sin(2*f0)*np.sum(self.R3_cross*rand_cross, axis=1)

        return R1, R2, R3

    def isgwb_aet_strain_response(self, f0):

        '''
        Calculate the detector transfer function functions to an isotropic SGWB non-polarized using AET TDI
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

        R1, R2 and R3   :   complex
            Antenna Patterns for the three channels, integrated over sky direction and averaged over polarization.
        '''

        ## check if the directional pattern functions exist
        try:
            self.R1_plus
        except:
            self.R_sky_response(f0)

        npix = hp.nside2npix(self.params['nside'])

        # Assign random complex amplitudes for each pixel
        rand_plus  = np.random.standard_normal(size=(f0.size, npix)) + 1j*np.random.standard_normal(size=(f0.size, npix))
        rand_cross = np.random.standard_normal(size=(f0.size, npix)) + 1j*np.random.standard_normal(size=(f0.size, npix))

        ## Initlize arrays for the detector reponse
        ## These are XYZ TDI Channels!
        R1 = np.zeros((f0.size, 2), dtype='complex')
        R2 = np.zeros((f0.size, 2), dtype='complex')
        R3 = np.zeros((f0.size, 2), dtype='complex')

        RA_plus =  (1/3)*(2*self.R1_plus - self.R2_plus - self.R3_plus)
        RE_plus =  (1/np.sqrt(3))*(self.R3_plus - self.R2_plus)
        RZ_plus =  (1/3)*(self.R1_plus + self.R2_plus + self.R3_plus)

        RA_cross =  (1/3)*(2*self.R1_cross - self.R2_cross - self.R3_cross)
        RE_cross =  (1/np.sqrt(3))*(self.R3_cross - self.R2_cross)
        RZ_cross =  (1/3)*(self.R1_cross + self.R2_cross + self.R3_cross)


        ## dot with the phases and calculate the pattern functions
        ## The factor of sin(2*f0) comes because there are XYZ channels
        ## rather than being michelson channels
        R1[:, 0], R1[:, 1] = 2*np.sin(2*f0)*np.sum(RA_plus*rand_plus, axis=1), 2*np.sin(2*f0)*np.sum(RA_cross*rand_cross, axis=1)
        R2[:, 0], R2[:, 1] = 2*np.sin(2*f0)*np.sum(RE_plus*rand_plus, axis=1), 2*np.sin(2*f0)*np.sum(RE_cross*rand_cross, axis=1)
        R3[:, 0], R3[:, 1] = 2*np.sin(2*f0)*np.sum(RZ_plus*rand_plus, axis=1), 2*np.sin(2*f0)*np.sum(RZ_cross*rand_cross, axis=1)

        return R1, R2, R3


    def asgwb_xyz_strain_response(self, f0):

        '''
        Calculate the detector transfer function functions to an anisotropic SGWB non-polarized xyz tdi
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
        import pdb; pdb.set_trace()



        npix = hp.nside2npix(self.params['nside'])

        # Array of pixel indices
        pix_idx  = np.arange(npix)

        #Angular coordinates of pixel indcides
        theta, phi = hp.pix2ang(self.params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)

        # Area of each pixel in sq.radians
        dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])

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
            Fplus1 = 0.5*(Fplus_u*gammaU_plus - Fplus_v*gammaV_plus)*np.exp(-1j*f0[ii]*(udir + vdir)/np.sqrt(3))
            Fplus2 = 0.5*(Fplus_w*gammaW_plus - Fplus_u*gammaU_minus)*np.exp(-1j*f0[ii]*(-udir + vdir)/np.sqrt(3))
            Fplus3 = 0.5*(Fplus_v*gammaV_minus - Fplus_w*gammaW_minus)*np.exp(1j*f0[ii]*(vdir + wdir)/np.sqrt(3))

            ## Calculate Fcross
            Fcross1 = 0.5*(Fcross_u*gammaU_plus - Fcross_v*gammaV_plus)*np.exp(-1j*f0[ii]*(udir + vdir)/np.sqrt(3))
            Fcross2 = 0.5*(Fcross_w*gammaW_plus - Fcross_u*gammaU_minus)*np.exp(-1j*f0[ii]*(-udir + vdir)/np.sqrt(3))
            Fcross3 = 0.5*(Fcross_v*gammaV_minus - Fcross_w*gammaW_minus)*np.exp(1j*f0[ii]*(vdir + wdir)/np.sqrt(3))

            ## Calculate antenna patterns for the X, Y, Z channels.
            FXplus = 2*np.sin(2*f0[ii])*Fplus1
            FYplus = 2*np.sin(2*f0[ii])*Fplus2
            FZplus = 2*np.sin(2*f0[ii])*Fplus3

            FXcross = 2*np.sin(2*f0[ii])*Fcross1
            FYcross = 2*np.sin(2*f0[ii])*Fcross2
            FZcross = 2*np.sin(2*f0[ii])*Fcross3

            ## Detector response summed over polarization and integrated over sky direction
            R1[ii,0], R1[ii, 1] = np.sqrt(0.5/npix)*np.sum(FXplus*rand_plus[:, ii]), np.sqrt(0.5/npix)*np.sum(FXcross*rand_cross[:, ii])
            R2[ii,0], R2[ii, 1] = np.sqrt(0.5/npix)*np.sum(FYplus*rand_plus[:, ii]), np.sqrt(0.5/npix)*np.sum(FYcross*rand_cross[:, ii])
            R3[ii,0], R3[ii, 1] = np.sqrt(0.5/npix)*np.sum(FZplus*rand_plus[:, ii]), np.sqrt(0.5/npix)*np.sum(FZcross*rand_cross[:, ii])


        return R1, R2, R3

