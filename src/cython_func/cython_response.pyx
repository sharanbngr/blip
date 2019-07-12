from __future__ import division
import numpy as np
cimport numpy as np
cimport cython


def isgwb_mich_strain_response(object self, double [:] f0):

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

        cdef double[:] tt = np.linspace(-1, 1, 200)
        cdef double[:] pp = np.linspace(0, 2*np.pi, 200, endpoint=False)
        cdef double[:,:] ct, phi

        [ct, phi] = np.meshgrid(tt,pp)
        cdef double dct = ct[0, 1] - ct[0,0]
        cdef double dphi = phi[1,0] - phi[0,0]

        ## udir is just u.r, where r is the directional vector
        cdef double[:,:] udir = np.sqrt(1-ct*ct) * np.sin(phi + np.pi/6)
        cdef double[:,:] vdir = np.sqrt(1-ct*ct) * np.sin(phi - np.pi/6)
        cdef double[:,:] wdir = vdir - udir

        # Initlize arrays for the detector reponse
        cdef complex[:,:] R1 = np.zeros((f0.size, 2), dtype='complex')
        cdef complex[:,:] R2 = np.zeros((f0.size, 2), dtype='complex')
        cdef complex[:,:] R3 = np.zeros((f0.size, 2), dtype='complex')


        ## response function u x u : eplus
        ##  Fplus_u = (u x u):eplus
        cdef double[:,:] Fplus_u   = (1/4*(1-ct*ct) + 1/2*(ct*ct)*(np.cos(phi))**2 - \
                        np.sqrt(3/16)*np.sin(2*phi)*(1+ct*ct)  + \
                            0.5*((np.cos(phi))**2 - ct*ct))
        
        cdef double[:,:] Fplus_v   = (1/4*(1-ct*ct) + 1/2*(ct*ct)*(np.cos(phi))**2 + \
                        np.sqrt(3/16)*np.sin(2*phi)*(1+ct*ct) + \
                            0.5*((np.cos(phi))**2 - ct*ct))

        cdef double[:,:] Fplus_w   = (1 - (1+ct*ct)*(np.cos(phi))**2)

        ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross
        cdef double[:,:] Fcross_u  = - ct * (np.sin(2*phi + np.pi/3))
        cdef double[:,:] Fcross_v  = - ct * (np.sin(2*phi - np.pi/3))
        cdef double[:,:] Fcross_w   = ct*np.sin(2*phi)


        cdef complex[:,:] gammaU_plus,gammaV_plus,gammaW_plus,gammaU_minus,gammaV_minus,gammaW_minus
        cdef complex[:,:] Fplus1,Fplus2,Fplus3,Fcross1,Fcross2,Fcross3, rand_plus, rand_cross

        cdef int numfreqs = f0.size
        cdef int npix = wdir.size



        # Calculate the detector response for each frequency
        for ii in range(numfreqs):

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
        


            ## Michelson antenna patterns
            ## Calculate Fplus
            Fplus1 = 0.5*(Fplus_u*gammaU_plus - Fplus_v*gammaV_plus)
            Fplus2 = 0.5*(Fplus_w*gammaW_plus - Fplus_u*gammaU_minus)*np.exp(2j*f0[ii]*udir)
            Fplus3 = 0.5*(Fplus_v*gammaV_minus - Fplus_w*gammaW_minus)*np.exp(2j*f0[ii]*vdir)

            ## Calculate Fcross
            Fcross1 = 0.5*(Fcross_u*gammaU_plus - Fcross_v*gammaV_plus)
            Fcross2 = 0.5*(Fcross_w*gammaW_plus - Fcross_u*gammaU_minus)*np.exp(2j*f0[ii]*udir)
            Fcross3 = 0.5*(Fcross_v*gammaV_minus - Fcross_w*gammaW_minus)*np.exp(2j*f0[ii]*vdir)

            rand_plus = np.random.normal(size=Fplus1.shape) + 1j* np.random.normal(size=Fplus1.shape)
            rand_cross = np.random.normal(size=Fplus1.shape) + 1j* np.random.normal(size=Fplus1.shape)

            
 
            ## Detector response summed over polarization and integrated over sky direction
            R1[ii, 0], R1[ii, 1] = np.sqrt(0.5/npix)*np.sum(Fplus1*rand_plus), np.sqrt(0.5/npix)*np.sum(Fcross1*rand_cross) 
            R2[ii, 0], R2[ii, 1] = np.sqrt(0.5/npix)*np.sum(Fplus2*rand_plus), np.sqrt(0.5/npix)*np.sum(Fcross2*rand_cross) 
            R3[ii, 0], R3[ii, 1] = np.sqrt(0.5/npix)*np.sum(Fplus3*rand_plus), np.sqrt(0.5/npix)*np.sum(Fcross3*rand_cross) 

  

        return R1, R2, R3