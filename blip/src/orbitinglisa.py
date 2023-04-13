import numpy as np
from scipy.special import lpmn
import healpy as hp

class orbitinglisa():

    '''
    Module containing geometry methods accounting for an orbiting satellite. The methods here include a base orbit, a single doppler channel, for 
    the three michelson channels or for the AET TDI channels and calculation of noise power spectra for various channel combinations. 
    '''

    
    def lisa_orbits(self, tsegmid):
    
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

        ## Semimajor axis in m
        a = 1.496e11
   
        
        ## Alpha and beta phases allow for changing of initial satellite orbital phases; default initial conditions are alphaphase=betaphase=0.
        betaphase = 0
        alphaphase = 0
        
        ## Orbital angle alpha(t)
        at = (2*np.pi/31557600)*tsegmid + alphaphase
        
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
        
        
    def orbiting_doppler_response(self, f0, theta, phi, tsegmid):
        
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
        
        self.rs1, self.rs2, self.rs3 = self.lisa_orbits(tsegmid)

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


    def orbiting_michelson_response(self, f0, theta, phi, tsegmid): 

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
        
        self.rs1, self.rs2, self.rs3 = self.lisa_orbits(tsegmid)

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

    def orbiting_aet_response(self, f0, theta, phi, tsegmid): 



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

        self.rs1, self.rs2, self.rs3 = self.lisa_orbits(tsegmid)

        R1plus, R1cross, R2plus, R2cross, R3plus, R3cross  = self.michelson_response(f0, theta, phi, tsegmid)
        

        ## Calculate antenna patterns for the A, E and T channels
        RAplus = (2/3)*np.sin(2*f0)*(2*R1plus - R2plus - R3plus)
        REplus = (2/np.sqrt(3))*np.sin(2*f0)*(R3plus - R2plus)
        RTplus = (1/3)*np.sin(2*f0)*(R1plus + R3plus + R2plus)

        RAcross = (2/3)*np.sin(2*f0)*(2*R1cross - R2cross - R3cross)
        REcross = (2/np.sqrt(3))*np.sin(2*f0)*(R3cross - R2cross)
        RTcross = (1/3)*np.sin(2*f0)*(R1cross + R3cross + R2cross)

        return RAplus, RAcross, REplus, REcross, RTplus, RTcross
    
    def isgwb_omich_response(self, f0, tsegmid):

        '''
        Calcualte the Antenna pattern/ detector transfer function functions for an orbiting LISA to an isotropic SGWB using basic michelson
        channels. Note that since this is the response to an isotropic background, the response function is integrated
        over sky direction and averaged over polarization. The angular integral is a linear and rectangular in the
        cos(theta) and phi space.  Note also that f0 is (pi*L*f)/c and is input as an array

        

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)
        
        tsegmid  :  array
            A numpy array of the midpoints for each time integration segment.
            
        rs1, rs2, rs3  :  array
            Satellite position vectors.
    

        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarization, evaluated at the midpoint of each time segment.
        '''
                
        self.rs1, self.rs2, self.rs3 = self.lisa_orbits(tsegmid)

        ## Setup healpix map
        npix = hp.nside2npix(self.params['nside'])

        # Array of pixel indices
        pix_idx  = np.arange(npix)

        #Angular coordinates of pixel indcides
        theta, phi = hp.pix2ang(self.params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)
        stheta = np.sqrt(1-ctheta**2)

        # Area of each pixel in sq.radians
        dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])

        ## Indices of midpoints array
        timeindices = np.arange(len(tsegmid))

        # Initlize arrays for the detector reponse
        R1 = np.zeros((len(timeindices),f0.size))
        R2 = np.zeros((len(timeindices),f0.size))
        R3 = np.zeros((len(timeindices),f0.size))

        ## arm vectors
        uvec = self.rs2 - self.rs1
        vvec = self.rs3 - self.rs1
        wvec = self.rs3 - self.rs2

        ## Calculate arm lengths
        Lu = np.sqrt(np.sum(uvec*uvec, axis=0))
        Lv = np.sqrt(np.sum(vvec*vvec, axis=0))
        Lw = np.sqrt(np.sum(wvec*wvec, axis=0))

        ## Define x/y/z for each satellite at all times
        x1, y1, z1 = self.rs1[0, :], self.rs1[1, :], self.rs1[2, :] 
        x2, y2, z2 = self.rs2[0, :], self.rs2[1, :], self.rs2[2, :] 
        x3, y3, z3 = self.rs3[0, :], self.rs3[1, :], self.rs3[2, :] 
        
        udir = np.tensordot((x2-x1)/Lu, np.cos(phi)*stheta, axes=0) + np.tensordot((y2-y1)/Lu, np.cos(phi)*stheta, axes=0) + \
                            np.tensordot((z2-z1)/Lu, ctheta, axes = 0)

        vdir = np.tensordot((x3-x1)/Lv, np.cos(phi)*stheta, axes=0) + np.tensordot((y3-y1)/Lv, np.cos(phi)*stheta, axes=0) + \
                            np.tensordot((z3-z1)/Lv, ctheta, axes = 0)

        wdir = np.tensordot((x3-x2)/Lw, np.cos(phi)*stheta, axes=0) + np.tensordot((y3-y2)/Lw, np.cos(phi)*stheta, axes=0) + \
                            np.tensordot((z3-z2)/Lw, ctheta, axes = 0)


        ## Calculate 1/2(u x u):eplus
        Pcontract_u = 1/2*( (np.tensordot((x2-x1)/Lu, np.sin(phi), axes=0) - np.tensordot((y2-y1)/Lu, np.cos(phi), axes=0))**2 - \
                    (np.tensordot((x2-x1)/Lu, np.cos(phi)*ctheta, axes=0) + np.tensordot((y2-y1)/Lu, np.cos(phi)*ctheta, axes=0) - \
                    np.tensordot((z2-z1)/Lu, stheta, axes=0))**2  )

        Pcontract_v = 1/2*( (np.tensordot((x3-x1)/Lv, np.sin(phi), axes=0) - np.tensordot((y3-y1)/Lv, np.cos(phi), axes=0))**2 - \
                    (np.tensordot((x3-x1)/Lv, np.cos(phi)*ctheta, axes=0) + np.tensordot((y3-y1)/Lv, np.cos(phi)*ctheta, axes=0) - \
                    np.tensordot((z3-z1)/Lv, stheta, axes=0))**2  )

        Pcontract_w = 1/2*( (np.tensordot((x3-x2)/Lw, np.sin(phi), axes=0) - np.tensordot((y3-y2)/Lw, np.cos(phi), axes=0))**2 - \
                    (np.tensordot((x3-x2)/Lw, np.cos(phi)*ctheta, axes=0) + np.tensordot((y3-y2)/Lw, np.cos(phi)*ctheta, axes=0) - \
                    np.tensordot((z3-z2)/Lw, stheta, axes=0))**2  )                           


        Ccontract_u = (np.tensordot((x2-x1)/Lu, np.sin(phi), axes=0) - np.tensordot((y2-y1)/Lu, np.cos(phi), axes=0) ) * \
                     ( np.tensordot((x2-x1)/Lu, np.cos(phi)*ctheta, axes=0) + np.tensordot((y2-y1)/Lu, np.cos(phi)*ctheta, axes=0) - \
                        np.tensordot((z2-z1)/Lu, stheta, axes=0) )  

        Ccontract_v = (np.tensordot((x3-x1)/Lv, np.sin(phi), axes=0) - np.tensordot((y3-y1)/Lv, np.cos(phi), axes=0) ) * \
                     ( np.tensordot((x3-x1)/Lv, np.cos(phi)*ctheta, axes=0) + np.tensordot((y3-y1)/Lv, np.cos(phi)*ctheta, axes=0) - \
                        np.tensordot((z3-z1)/Lv, stheta, axes=0) ) 

        Ccontract_w = (np.tensordot((x3-x2)/Lw, np.sin(phi), axes=0) - np.tensordot((y3-y2)/Lw, np.cos(phi), axes=0) ) * \
                     ( np.tensordot((x3-x2)/Lw, np.cos(phi)*ctheta, axes=0) + np.tensordot((y3-y2)/Lw, np.cos(phi)*ctheta, axes=0) - \
                        np.tensordot((z3-z2)/Lw, stheta, axes=0) ) 



        # Calculate the detector response for each frequency
        for ii in range(0, f0.size):
            # Calculate GW transfer function for the michelson channels
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
            Fplus1 = (Fplus_u_p - Fplus_v_p)
            Fplus2 = (Fplus_w_p - Fplus_u_m)
            Fplus3 = (Fplus_v_m - Fplus_w_m)
    
            ## Calculate Fcross
            Fcross1 = (Fcross_u_p - Fcross_v_p)
            Fcross2 = (Fcross_w_p - Fcross_u_m)
            Fcross3 = (Fcross_v_m - Fcross_w_m)

            ## Detector response summed over polarization and integrated over sky direction
            R1[:, ii] = dOmega/(8*np.pi)*np.sum((np.absolute(Fplus1))**2 + (np.absolute(Fcross1))**2, axis=1)
            R2[:, ii] = dOmega/(8*np.pi)*np.sum((np.absolute(Fplus2))**2 + (np.absolute(Fcross2))**2, axis=1)
            R3[:, ii] = dOmega/(8*np.pi)*np.sum((np.absolute(Fplus3))**2 + (np.absolute(Fcross3))**2, axis=1)
        
        ## Output detector response arrays; these can then be loaded instead of calculated for future analyses of the same data.
        np.savetxt('R1arrayMich.txt',R1)
        np.savetxt('R2arrayMich.txt',R2)
        np.savetxt('R3arrayMich.txt',R3)
        
        return R1, R2, R3

    def isgwb_oxyz_response(self, f0, tsegmid): 

        '''
        Calcualte the Antenna pattern/ detector transfer function functions to an isotropic SGWB using X, Y and Z TDI channels for an orbiting LISA. Note that since this is the response to an isotropic background, the response function is integrated over sky direction and averaged over polarozation. The angular integral is a linear and rectangular in the cos(theta) and phi space.  Note that f0 is (pi*L*f)/c and is input as an array

        

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)
            
        tsegmid  :  array
            A numpy array of the midpoints for each time integration segment.
            
        rs1, rs2, rs3  :  array
            Satellite position vectors.

        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarization, evaluated at the midpoint of each time segment.
        '''
        
        self.rs1, self.rs2, self.rs3 = self.lisa_orbits(tsegmid)

        ## Setup healpix map
        npix = hp.nside2npix(self.params['nside'])

        # Array of pixel indices
        pix_idx  = np.arange(npix)

        #Angular coordinates of pixel indcides
        theta, phi = hp.pix2ang(self.params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)
        stheta = np.sqrt(1-ctheta**2)

        # Area of each pixel in sq.radians
        dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])

        ## Indices of midpoints array
        timeindices = np.arange(len(tsegmid))

        # Initlize arrays for the detector reponse
        R1 = np.zeros((len(timeindices),f0.size))
        R2 = np.zeros((len(timeindices),f0.size))
        R3 = np.zeros((len(timeindices),f0.size))

        ## arm vectors
        uvec = self.rs2 - self.rs1
        vvec = self.rs3 - self.rs1
        wvec = self.rs3 - self.rs2

        ## Calculate arm lengths
        Lu = np.sqrt(np.sum(uvec*uvec, axis=0))
        Lv = np.sqrt(np.sum(vvec*vvec, axis=0))
        Lw = np.sqrt(np.sum(wvec*wvec, axis=0))

        ## Define x/y/z for each satellite at all times
        x1, y1, z1 = self.rs1[0, :], self.rs1[1, :], self.rs1[2, :] 
        x2, y2, z2 = self.rs2[0, :], self.rs2[1, :], self.rs2[2, :] 
        x3, y3, z3 = self.rs3[0, :], self.rs3[1, :], self.rs3[2, :] 
        
        udir = np.tensordot((x2-x1)/Lu, np.cos(phi)*stheta, axes=0) + np.tensordot((y2-y1)/Lu, np.cos(phi)*stheta, axes=0) + \
                            np.tensordot((z2-z1)/Lu, ctheta, axes = 0)

        vdir = np.tensordot((x3-x1)/Lv, np.cos(phi)*stheta, axes=0) + np.tensordot((y3-y1)/Lv, np.cos(phi)*stheta, axes=0) + \
                            np.tensordot((z3-z1)/Lv, ctheta, axes = 0)

        wdir = np.tensordot((x3-x2)/Lw, np.cos(phi)*stheta, axes=0) + np.tensordot((y3-y2)/Lw, np.cos(phi)*stheta, axes=0) + \
                            np.tensordot((z3-z2)/Lw, ctheta, axes = 0)


        ## Calculate 1/2(u x u):eplus
        Pcontract_u = 1/2*( (np.tensordot((x2-x1)/Lu, np.sin(phi), axes=0) - np.tensordot((y2-y1)/Lu, np.cos(phi), axes=0))**2 - \
                    (np.tensordot((x2-x1)/Lu, np.cos(phi)*ctheta, axes=0) + np.tensordot((y2-y1)/Lu, np.cos(phi)*ctheta, axes=0) - \
                    np.tensordot((z2-z1)/Lu, stheta, axes=0))**2  )

        Pcontract_v = 1/2*( (np.tensordot((x3-x1)/Lv, np.sin(phi), axes=0) - np.tensordot((y3-y1)/Lv, np.cos(phi), axes=0))**2 - \
                    (np.tensordot((x3-x1)/Lv, np.cos(phi)*ctheta, axes=0) + np.tensordot((y3-y1)/Lv, np.cos(phi)*ctheta, axes=0) - \
                    np.tensordot((z3-z1)/Lv, stheta, axes=0))**2  )

        Pcontract_w = 1/2*( (np.tensordot((x3-x2)/Lw, np.sin(phi), axes=0) - np.tensordot((y3-y2)/Lw, np.cos(phi), axes=0))**2 - \
                    (np.tensordot((x3-x2)/Lw, np.cos(phi)*ctheta, axes=0) + np.tensordot((y3-y2)/Lw, np.cos(phi)*ctheta, axes=0) - \
                    np.tensordot((z3-z2)/Lw, stheta, axes=0))**2  )                           


        Ccontract_u = (np.tensordot((x2-x1)/Lu, np.sin(phi), axes=0) - np.tensordot((y2-y1)/Lu, np.cos(phi), axes=0) ) * \
                     ( np.tensordot((x2-x1)/Lu, np.cos(phi)*ctheta, axes=0) + np.tensordot((y2-y1)/Lu, np.cos(phi)*ctheta, axes=0) - \
                        np.tensordot((z2-z1)/Lu, stheta, axes=0) )  

        Ccontract_v = (np.tensordot((x3-x1)/Lv, np.sin(phi), axes=0) - np.tensordot((y3-y1)/Lv, np.cos(phi), axes=0) ) * \
                     ( np.tensordot((x3-x1)/Lv, np.cos(phi)*ctheta, axes=0) + np.tensordot((y3-y1)/Lv, np.cos(phi)*ctheta, axes=0) - \
                        np.tensordot((z3-z1)/Lv, stheta, axes=0) ) 

        Ccontract_w = (np.tensordot((x3-x2)/Lw, np.sin(phi), axes=0) - np.tensordot((y3-y2)/Lw, np.cos(phi), axes=0) ) * \
                     ( np.tensordot((x3-x2)/Lw, np.cos(phi)*ctheta, axes=0) + np.tensordot((y3-y2)/Lw, np.cos(phi)*ctheta, axes=0) - \
                        np.tensordot((z3-z2)/Lw, stheta, axes=0) ) 



        # Calculate the detector response for each frequency
        for ii in range(0, f0.size):
            # Calculate GW transfer function for the michelson channels
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
            Fplus1 = (Fplus_u_p - Fplus_v_p)
            Fplus2 = (Fplus_w_p - Fplus_u_m)
            Fplus3 = (Fplus_v_m - Fplus_w_m)
    
            ## Calculate Fcross
            Fcross1 = (Fcross_u_p - Fcross_v_p)
            Fcross2 = (Fcross_w_p - Fcross_u_m)
            Fcross3 = (Fcross_v_m - Fcross_w_m)        

            ## Calculate antenna patterns for the X, Y, Z channels.
            FXplus = 2*np.sin(2*f0[ii])*Fplus1
            FYplus = 2*np.sin(2*f0[ii])*Fplus2
            FZplus = 2*np.sin(2*f0[ii])*Fplus3

            FXcross = 2*np.sin(2*f0[ii])*Fcross1
            FYcross = 2*np.sin(2*f0[ii])*Fcross2
            FZcross = 2*np.sin(2*f0[ii])*Fcross3


            ## Detector response for the TDI Channels, summed over polarization
            ## and integrated over sky direction
            R1[:, ii] = dOmega/(8*np.pi)*np.sum((np.absolute(FXplus))**2 + (np.absolute(FXcross))**2, axis=1)
            R2[:, ii] = dOmega/(8*np.pi)*np.sum((np.absolute(FYplus))**2 + (np.absolute(FYcross))**2, axis=1)
            R3[:, ii] = dOmega/(8*np.pi)*np.sum((np.absolute(FZplus))**2 + (np.absolute(FZcross))**2, axis=1)

        ## Output detector response arrays; these can then be loaded instead of calculated for future analyses of the same data.
        np.savetxt('R1arrayXYZ.txt',R1)
        np.savetxt('R2arrayXYZ.txt',R2)
        np.savetxt('R3arrayXYZ.txt',R3)
        
        return R1, R2, R3

    def isgwb_oaet_response(self, f0, tsegmid): 

        '''
        Calcualte the Antenna pattern/ detector transfer function functions to an isotropic SGWB using A, E and T TDI channels. Note that since this is the response to an isotropic background, the response function is integrated over sky direction and averaged over polarization. The angular integral is a linear and rectangular in the cos(theta) and phi space.  Note that f0 is (pi*L*f)/c and is input as an array.

        

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)
        
        tsegmid  :  array
            A numpy array of the midpoints for each time integration segment.
        
        rs1, rs2, rs3  :  array
            Satellite position vectors.
          
        Returns
        ---------

        R1, R2 and R3   :   arrays
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarizationevaluated at the midpoint of each time segment.
        '''

        self.rs1, self.rs2, self.rs3 = self.lisa_orbits(tsegmid)

        ## Setup healpix map
        npix = hp.nside2npix(self.params['nside'])

        # Array of pixel indices
        pix_idx  = np.arange(npix)

        #Angular coordinates of pixel indcides
        theta, phi = hp.pix2ang(self.params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)
        stheta = np.sqrt(1-ctheta**2)

        # Area of each pixel in sq.radians
        dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])

        ## Indices of midpoints array
        timeindices = np.arange(len(tsegmid))

        # Initlize arrays for the detector reponse
        R1 = np.zeros((len(timeindices),f0.size))
        R2 = np.zeros((len(timeindices),f0.size))
        R3 = np.zeros((len(timeindices),f0.size))

        ## arm vectors
        uvec = self.rs2 - self.rs1
        vvec = self.rs3 - self.rs1
        wvec = self.rs3 - self.rs2

        ## Calculate arm lengths
        Lu = np.sqrt(np.sum(uvec*uvec, axis=0))
        Lv = np.sqrt(np.sum(vvec*vvec, axis=0))
        Lw = np.sqrt(np.sum(wvec*wvec, axis=0))

        ## Define x/y/z for each satellite at all times
        x1, y1, z1 = self.rs1[0, :], self.rs1[1, :], self.rs1[2, :] 
        x2, y2, z2 = self.rs2[0, :], self.rs2[1, :], self.rs2[2, :] 
        x3, y3, z3 = self.rs3[0, :], self.rs3[1, :], self.rs3[2, :] 
        
        udir = np.tensordot((x2-x1)/Lu, np.cos(phi)*stheta, axes=0) + np.tensordot((y2-y1)/Lu, np.cos(phi)*stheta, axes=0) + \
                            np.tensordot((z2-z1)/Lu, ctheta, axes = 0)

        vdir = np.tensordot((x3-x1)/Lv, np.cos(phi)*stheta, axes=0) + np.tensordot((y3-y1)/Lv, np.cos(phi)*stheta, axes=0) + \
                            np.tensordot((z3-z1)/Lv, ctheta, axes = 0)

        wdir = np.tensordot((x3-x2)/Lw, np.cos(phi)*stheta, axes=0) + np.tensordot((y3-y2)/Lw, np.cos(phi)*stheta, axes=0) + \
                            np.tensordot((z3-z2)/Lw, ctheta, axes = 0)


        ## Calculate 1/2(u x u):eplus
        Pcontract_u = 1/2*( (np.tensordot((x2-x1)/Lu, np.sin(phi), axes=0) - np.tensordot((y2-y1)/Lu, np.cos(phi), axes=0))**2 - \
                    (np.tensordot((x2-x1)/Lu, np.cos(phi)*ctheta, axes=0) + np.tensordot((y2-y1)/Lu, np.cos(phi)*ctheta, axes=0) - \
                    np.tensordot((z2-z1)/Lu, stheta, axes=0))**2  )

        Pcontract_v = 1/2*( (np.tensordot((x3-x1)/Lv, np.sin(phi), axes=0) - np.tensordot((y3-y1)/Lv, np.cos(phi), axes=0))**2 - \
                    (np.tensordot((x3-x1)/Lv, np.cos(phi)*ctheta, axes=0) + np.tensordot((y3-y1)/Lv, np.cos(phi)*ctheta, axes=0) - \
                    np.tensordot((z3-z1)/Lv, stheta, axes=0))**2  )

        Pcontract_w = 1/2*( (np.tensordot((x3-x2)/Lw, np.sin(phi), axes=0) - np.tensordot((y3-y2)/Lw, np.cos(phi), axes=0))**2 - \
                    (np.tensordot((x3-x2)/Lw, np.cos(phi)*ctheta, axes=0) + np.tensordot((y3-y2)/Lw, np.cos(phi)*ctheta, axes=0) - \
                    np.tensordot((z3-z2)/Lw, stheta, axes=0))**2  )                           


        Ccontract_u = (np.tensordot((x2-x1)/Lu, np.sin(phi), axes=0) - np.tensordot((y2-y1)/Lu, np.cos(phi), axes=0) ) * \
                     ( np.tensordot((x2-x1)/Lu, np.cos(phi)*ctheta, axes=0) + np.tensordot((y2-y1)/Lu, np.cos(phi)*ctheta, axes=0) - \
                        np.tensordot((z2-z1)/Lu, stheta, axes=0) )  

        Ccontract_v = (np.tensordot((x3-x1)/Lv, np.sin(phi), axes=0) - np.tensordot((y3-y1)/Lv, np.cos(phi), axes=0) ) * \
                     ( np.tensordot((x3-x1)/Lv, np.cos(phi)*ctheta, axes=0) + np.tensordot((y3-y1)/Lv, np.cos(phi)*ctheta, axes=0) - \
                        np.tensordot((z3-z1)/Lv, stheta, axes=0) ) 

        Ccontract_w = (np.tensordot((x3-x2)/Lw, np.sin(phi), axes=0) - np.tensordot((y3-y2)/Lw, np.cos(phi), axes=0) ) * \
                     ( np.tensordot((x3-x2)/Lw, np.cos(phi)*ctheta, axes=0) + np.tensordot((y3-y2)/Lw, np.cos(phi)*ctheta, axes=0) - \
                        np.tensordot((z3-z2)/Lw, stheta, axes=0) ) 



        # Calculate the detector response for each frequency
        for ii in range(0, f0.size):
            # Calculate GW transfer function for the michelson channels
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
            Fplus1 = (Fplus_u_p - Fplus_v_p)
            Fplus2 = (Fplus_w_p - Fplus_u_m)
            Fplus3 = (Fplus_v_m - Fplus_w_m)
    
            ## Calculate Fcross
            Fcross1 = (Fcross_u_p - Fcross_v_p)
            Fcross2 = (Fcross_w_p - Fcross_u_m)
            Fcross3 = (Fcross_v_m - Fcross_w_m)        
                
    
            ## Calculate antenna patterns for the A, E and T channels -  We are switiching to doppler channel.
            FAplus = (1/3)*np.sin(2*f0[ii])*(2*Fplus1 - Fplus2 - Fplus3)
            FEplus = (1/np.sqrt(3))*np.sin(2*f0[ii])*(Fplus3 - Fplus2)
            FTplus = (1/3)*np.sin(2*f0[ii])*(Fplus1 + Fplus3 + Fplus2)
    
            FAcross = (1/3)*np.sin(2*f0[ii])*(2*Fcross1 - Fcross2 - Fcross3)
            FEcross = (1/np.sqrt(3))*np.sin(2*f0[ii])*(Fcross3 - Fcross2)
            FTcross = (1/3)*np.sin(2*f0[ii])*(Fcross1 + Fcross3 + Fcross2)
    
            ## Detector response for the TDI Channels, summed over polarization
            ## and integrated over sky direction
            R1[:, ii] = dOmega/(8*np.pi)*np.sum((np.absolute(FAplus))**2 + (np.absolute(FAcross))**2, axis=1)
            R2[:, ii] = dOmega/(8*np.pi)*np.sum((np.absolute(FEplus))**2 + (np.absolute(FEcross))**2, axis=1)
            R3[:, ii] = dOmega/(8*np.pi)*np.sum((np.absolute(FTplus))**2 + (np.absolute(FTcross))**2, axis=1)


        ## Output detector response arrays; these can then be loaded instead of calculated for future analyses of the same data.
        np.savetxt('R1arrayAET.txt',R1)
        np.savetxt('R2arrayAET.txt',R2)
        np.savetxt('R3arrayAET.txt',R3)
                
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
            gammaU    =    1/2 * (np.sinc((f0[ii])*(1-udir))*np.exp(-1j*f0[ii]*(3+udir)) + \
                             np.sinc((f0[ii])*(1+udir))*np.exp(-1j*f0[ii]*(1+udir)))

            gammaV    =    1/2 * (np.sinc((f0[ii])*(1-vdir))*np.exp(-1j*f0[ii]*(3+vdir)) + \
                             np.sinc((f0[ii])*(1+vdir))*np.exp(-1j*f0[ii]*(1+vdir)))

            gammaW    =    1/2 * (np.sinc((f0[ii])*(1-wdir))*np.exp(-1j*f0[ii]*(3+wdir)) + \
                             np.sinc((f0[ii])*(1+wdir))*np.exp(-1j*f0[ii]*(1+wdir)))

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
            Fcross_w  = 1/2*ct*np.sin(2*phi)*gammaW


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