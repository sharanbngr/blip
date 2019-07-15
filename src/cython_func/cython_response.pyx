from __future__ import division
import numpy as np
#cimport numpy as np
cimport cython

cdef extern from "gsl/gsl_sf_trig.h":

    double gsl_sf_sinc(double)


cdef double sinc_gsl(double x):
    
    return gsl_sf_sinc(x)

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

        cdef double pi_val = np.pi
        cdef double[:] ct = np.linspace(-1, 1, 200)
        cdef double[:] phi = np.linspace(0, 2*pi_val, 210, endpoint=False)

        cdef double dct = ct[1] - ct[0]
        cdef double dphi = phi[1] - phi[0]

        cdef int numfreqs = f0.size
        cdef int numtheta =  ct.size
        cdef int numphi =  phi.size
        cdef int ii, jj, kk

        cdef double[:, :] udir = np.zeros((numphi, numtheta))
        cdef double[:, :] vdir = np.zeros((numphi, numtheta)) 
        cdef double[:, :] wdir = np.zeros((numphi, numtheta))

        cdef double[:,:] Fplus_u = np.zeros((numphi, numtheta))
        cdef double[:,:] Fplus_v = np.zeros((numphi, numtheta))
        cdef double[:,:] Fplus_w = np.zeros((numphi, numtheta))

        cdef double[:,:] Fcross_u = np.zeros((numphi, numtheta))
        cdef double[:,:] Fcross_v = np.zeros((numphi, numtheta))
        cdef double[:,:] Fcross_w = np.zeros((numphi, numtheta))

        cdef norm = np.sqrt(0.5/udir.size)

        # Initlize arrays for the detector reponse
        cdef complex[:,:] R1 = np.zeros((f0.size, 2), dtype='complex')
        cdef complex[:,:] R2 = np.zeros((f0.size, 2), dtype='complex')
        cdef complex[:,:] R3 = np.zeros((f0.size, 2), dtype='complex')

        cdef complex gammaU_plus,gammaV_plus,gammaW_plus,gammaU_minus,gammaV_minus,gammaW_minus
        cdef complex Fplus1,Fplus2,Fplus3,Fcross1,Fcross2,Fcross3, rand_plus, rand_cross

        for ii in range(numphi):
            for jj in range(numtheta):
                udir[ii, jj] = np.sqrt(1-ct[jj]**2) * np.sin(phi[ii] + pi_val/6)
                vdir[ii, jj] = np.sqrt(1-ct[jj]**2) * np.sin(phi[ii] - pi_val/6)
                wdir[ii, jj] = vdir[ii, jj]  - udir[ii, jj]

                ##  Fplus_u = (u x u):eplus
                Fplus_u[ii, jj]= (1/4*(1-ct[jj]**2) + 1/2*(ct[jj]**2)*(np.cos(phi[ii]))**2 - \
                        np.sqrt(3/16)*np.sin(2*phi[ii])*(1+ct[jj]**2)  + \
                            0.5*((np.cos(phi[ii]))**2 - ct[jj]**2))
        
                Fplus_v[ii, jj]   = (1/4*(1-ct[jj]**2) + 1/2*(ct[jj]**2)*(np.cos(phi[ii]))**2 + \
                        np.sqrt(3/16)*np.sin(2*phi[ii])*(1+ct[jj]**2) + \
                            0.5*((np.cos(phi[ii]))**2 - ct[jj]**2))

                Fplus_w[ii, jj]    = (1 - (1+ct[jj]**2)*(np.cos(phi[ii]))**2)

                ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross
                Fcross_u[ii, jj]   = - ct[jj] * (np.sin(2*phi[ii] + pi_val/3))
                Fcross_v[ii, jj]   = - ct[jj] * (np.sin(2*phi[ii] - pi_val/3))
                Fcross_w[ii, jj]   = ct[jj] * np.sin(2*phi[ii])

        # Calculate the detector response for each frequency
        for ii in range(numfreqs):

            print str(ii) + ' of ' + str(numfreqs)
            for jj in range(numphi):
                for kk in range(numtheta):

                    # Calculate GW transfer function for the michelson channels
                    gammaU_plus    =    1/2 * (sinc_gsl((f0[ii])*(1 - udir[jj, kk])/pi_val)*np.exp(-1j*f0[ii]*(3+udir[jj, kk])) + \
                                 sinc_gsl((f0[ii])*(1 + udir[jj, kk])/pi_val)*np.exp(-1j*f0[ii]*(1+udir[jj, kk])))

                    gammaV_plus    =    1/2 * (sinc_gsl((f0[ii])*(1 - vdir[jj, kk])/pi_val)*np.exp(-1j*f0[ii]*(3+vdir[jj,kk])) + \
                                 sinc_gsl((f0[ii])*(1 + vdir[jj,kk])/pi_val)*np.exp(-1j*f0[ii]*(1+vdir[jj,kk])))

                    gammaW_plus    =    1/2 * (sinc_gsl((f0[ii])*(1 - wdir[jj,kk])/pi_val)*np.exp(-1j*f0[ii]*(3+wdir[jj,kk])) + \
                                 sinc_gsl((f0[ii])*(1 + wdir[jj,kk])/pi_val)*np.exp(-1j*f0[ii]*(1+wdir[jj,kk])))
            
                    # Calculate GW transfer function for the michelson channels
                    gammaU_minus    =    1/2 * (sinc_gsl((f0[ii])*(1 + udir[jj,kk])/pi_val)*np.exp(-1j*f0[ii]*(3 - udir[jj,kk])) + \
                                 sinc_gsl((f0[ii])*(1 - udir[jj,kk])/pi_val)*np.exp(-1j*f0[ii]*(1 - udir[jj,kk])))

                    gammaV_minus    =    1/2 * (sinc_gsl((f0[ii])*(1 + vdir[jj,kk])/pi_val)*np.exp(-1j*f0[ii]*(3 - vdir[jj,kk])) + \
                                 sinc_gsl((f0[ii])*(1 - vdir[jj,kk])/pi_val)*np.exp(-1j*f0[ii]*(1 - vdir[jj,kk])))

                    gammaW_minus    =    1/2 * (sinc_gsl((f0[ii])*(1 + wdir[jj,kk])/pi_val)*np.exp(-1j*f0[ii]*(3 - wdir[jj,kk])) + \
                                 sinc_gsl((f0[ii])*(1 - wdir[jj,kk])/np.pi)*np.exp(-1j*f0[ii]*(1 - wdir[jj,kk])))
        


                    ## Michelson antenna patterns: Calculate Fplus
                    Fplus1 = 0.5*(Fplus_u[jj, kk]*gammaU_plus - Fplus_v[jj, kk]*gammaV_plus)
                    Fplus2 = 0.5*(Fplus_w[jj, kk]*gammaW_plus - Fplus_u[jj, kk]*gammaU_minus)*np.exp(2j*f0[ii]*udir[jj,kk])
                    Fplus3 = 0.5*(Fplus_v[jj, kk]*gammaV_minus - Fplus_w[jj, kk]*gammaW_minus)*np.exp(2j*f0[ii]*vdir[jj,kk])

                    ## Michelson antenna patterns: Calculate Fcross
                    Fcross1 = 0.5*(Fcross_u[jj, kk]*gammaU_plus - Fcross_v[jj, kk]*gammaV_plus)
                    Fcross2 = 0.5*(Fcross_w[jj, kk]*gammaW_plus - Fcross_u[jj, kk]*gammaU_minus)*np.exp(2j*f0[ii]*udir[jj,kk])
                    Fcross3 = 0.5*(Fcross_v[jj, kk]*gammaV_minus - Fcross_w[jj, kk]*gammaW_minus)*np.exp(2j*f0[ii]*vdir[jj,kk])

                    rand_plus = np.random.normal() + 1j*np.random.normal()
                    rand_cross = np.random.normal() + 1j* np.random.normal()

                    ## Detector response summed over polarization and integrated over sky direction
                    R1[ii, 0], R1[ii, 1] = R1[ii, 0] + norm*Fplus1*rand_plus, R1[ii, 1] + norm*Fcross1*rand_cross
                    R2[ii, 0], R2[ii, 1] = R2[ii, 0] + norm*Fplus2*rand_plus, R2[ii, 1] + norm*Fcross2*rand_cross 
                    R3[ii, 0], R3[ii, 1] = R3[ii, 0] + norm*Fplus3*rand_plus, R3[ii, 1] + norm*Fcross3*rand_cross 

  

        return R1, R2, R3