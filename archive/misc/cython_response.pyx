# cython: profile=True
from __future__ import division
import numpy as np
cimport cython

cdef extern from "gsl/gsl_sf_trig.h":

    double gsl_sf_sinc(double)
    double gsl_sf_sin(double)
    double gsl_sf_cos(double)
    
cdef extern from "math.h":
   double sqrt(double)

#cdef extern from "gsl/gsl_math.h":


cdef extern from "gsl/gsl_rng.h":
   ctypedef struct gsl_rng_type:
       pass
   ctypedef struct gsl_rng:
       pass
   gsl_rng_type *gsl_rng_mt19937
   gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
  
cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

cdef extern from "gsl/gsl_randist.h":
   double gsl_ran_gaussian(gsl_rng * r,double)


### --------------- Methods calling GSL Functions ------------------------------------

cdef double unit_nrm_gsl():
    return gsl_ran_gaussian(r, 1.0)

cdef double sine_gsl(double x):
    return gsl_sf_sin(x)

cdef double cos_gsl(double x):
    return gsl_sf_cos(x)

cdef double sinc_gsl(double x):
    return gsl_sf_sinc(x)

  
cdef double sqrt_gsl(double n):
   return sqrt(n)

cdef complex cmp_exp_gsl(double x):
    return  gsl_sf_cos(x) + 1j*gsl_sf_sin(x)

def isgwb_mich_strain_response(object self, double[:] f0):

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

    
        tmids: Mid segments
        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarization.
        '''


        cdef double pi_val = 3.141592653589793238462
        cdef double[:] ct = np.linspace(-1, 1, 150)
        cdef double[:] phi = np.linspace(0, 2*pi_val, 150, endpoint=False)

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

        #cdef complex[:,:,:] rand_plus = np.zeros((numphi, numtheta, f0.size), dtype='complex')
        #cdef complex[:,:,:] rand_cross = np.zeros((numphi, numtheta, f0.size), dtype='complex')

        cdef norm = sqrt_gsl(0.5/udir.size)

        cdef complex exp_3f0, exp_f0, exp_u, exp_v
        cdef complex exp_u_minus, exp_v_minus, exp_w_minus
        cdef complex exp_u_plus, exp_v_plus, exp_w_plus 

        # Initlize arrays for the detector reponse
        cdef complex[:,:] R1 = np.zeros((f0.size, numphi, numtheta, 2), dtype='complex')
        cdef complex[:,:] R2 = np.zeros((f0.size, numphi, numtheta, 2), dtype='complex')
        cdef complex[:,:] R3 = np.zeros((f0.size, numphi, numtheta, 2), dtype='complex')

        cdef complex gammaU_plus,gammaV_plus,gammaW_plus,gammaU_minus,gammaV_minus,gammaW_minus
        cdef complex Fplus1,Fplus2,Fplus3,Fcross1,Fcross2,Fcross3

        for ii in range(numphi):
            for jj in range(numtheta):
                udir[ii, jj] = sqrt_gsl(1-ct[jj]**2) * sine_gsl(phi[ii] + pi_val/6)
                vdir[ii, jj] = sqrt_gsl(1-ct[jj]**2) * sine_gsl(phi[ii] - pi_val/6)
                wdir[ii, jj] = vdir[ii, jj]  - udir[ii, jj]

                ##  Fplus_u = (u x u):eplus
                Fplus_u[ii, jj]= (1/4*(1-ct[jj]**2) + 1/2*(ct[jj]**2)*(cos_gsl(phi[ii]))**2 - \
                        sqrt_gsl(3/16)*sine_gsl(2*phi[ii])*(1+ct[jj]**2)  + \
                            0.5*((cos_gsl(phi[ii]))**2 - ct[jj]**2))
        
                Fplus_v[ii, jj]   = (1/4*(1-ct[jj]**2) + 1/2*(ct[jj]**2)*(cos_gsl(phi[ii]))**2 + \
                        sqrt_gsl(3/16)*sine_gsl(2*phi[ii])*(1+ct[jj]**2) + \
                            0.5*((cos_gsl(phi[ii]))**2 - ct[jj]**2))

                Fplus_w[ii, jj]    = (1 - (1+ct[jj]**2)*(cos_gsl(phi[ii]))**2)

                ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross
                Fcross_u[ii, jj]   = - ct[jj] * (sine_gsl(2*phi[ii] + pi_val/3))
                Fcross_v[ii, jj]   = - ct[jj] * (sine_gsl(2*phi[ii] - pi_val/3))
                Fcross_w[ii, jj]   = ct[jj] * sine_gsl(2*phi[ii])


        for ii in range(numphi):
            for jj in range(numtheta):
                for kk in range(f0.size):
                    rand_plus[ii, jj, kk] = cmp_exp_gsl(2*pi_val*unit_nrm_gsl())
                    rand_cross[ii, jj, kk] = cmp_exp_gsl(2*pi_val*unit_nrm_gsl())

        # Calculate the detector response for each frequency
        for ii in range(numfreqs):
            exp_3f0 = cmp_exp_gsl(-3*f0[ii])
            exp_f0 = cmp_exp_gsl(-f0[ii])
            
            for jj in range(numphi):
                for kk in range(numtheta):

                    exp_u_minus = cmp_exp_gsl(-f0[ii]*udir[jj, kk])
                    exp_v_minus = cmp_exp_gsl(-f0[ii]*vdir[jj, kk])
                    exp_w_minus = cmp_exp_gsl(-f0[ii]*wdir[jj, kk])

                    exp_u_plus, exp_v_plus, exp_w_plus   = exp_u_minus.conjugate(), exp_v_minus.conjugate(), exp_w_minus.conjugate()

                    # Calculate GW transfer function for the michelson channels
                    gammaU_plus    =    1/2 * (sinc_gsl((f0[ii])*(1 - udir[jj, kk])/pi_val)*exp_3f0 + \
                                    sinc_gsl((f0[ii])*(1 + udir[jj, kk])/pi_val)*exp_3f0) * exp_u_minus

                    gammaV_plus    =    1/2 * (sinc_gsl((f0[ii])*(1 - vdir[jj, kk])/pi_val)*exp_3f0 + \
                                    sinc_gsl((f0[ii])*(1 + vdir[jj,kk])/pi_val)*exp_3f0) * exp_v_minus

                    gammaW_plus    =    1/2 * (sinc_gsl((f0[ii])*(1 - wdir[jj, kk])/pi_val)*exp_3f0 + \
                                    sinc_gsl((f0[ii])*(1 + wdir[jj,kk])/pi_val)*exp_3f0) * exp_w_minus
            
                    # Calculate GW transfer function for the michelson channels
                    gammaU_minus    =    1/2 * (sinc_gsl((f0[ii])*(1 + udir[jj,kk])/pi_val)*exp_3f0 + \
                                    sinc_gsl((f0[ii])*(1 - udir[jj,kk])/pi_val)*exp_3f0) * exp_u_plus

                    gammaV_minus    =    1/2 * (sinc_gsl((f0[ii])*(1 + vdir[jj,kk])/pi_val)*exp_3f0 + \
                                    sinc_gsl((f0[ii])*(1 - vdir[jj,kk])/pi_val)*exp_3f0) * exp_v_plus

                    gammaW_minus    =    1/2 * (sinc_gsl((f0[ii])*(1 + wdir[jj,kk])/pi_val)*exp_3f0 + \
                                    sinc_gsl((f0[ii])*(1 - wdir[jj,kk])/pi_val)*exp_3f0) * exp_w_plus
        
                    exp_u  = cmp_exp_gsl(2*f0[ii]*udir[jj,kk])
                    exp_v  = cmp_exp_gsl(2*f0[ii]*vdir[jj,kk])

                    ## Michelson antenna patterns: Calculate Fplus
                    Fplus1 = 0.5*(Fplus_u[jj, kk]*gammaU_plus - Fplus_v[jj, kk]*gammaV_plus)
                    Fplus2 = 0.5*(Fplus_w[jj, kk]*gammaW_plus - Fplus_u[jj, kk]*gammaU_minus)*exp_u
                    Fplus3 = 0.5*(Fplus_v[jj, kk]*gammaV_minus - Fplus_w[jj, kk]*gammaW_minus)*exp_v

                    ## Michelson antenna patterns: Calculate Fcross
                    Fcross1 = 0.5*(Fcross_u[jj, kk]*gammaU_plus - Fcross_v[jj, kk]*gammaV_plus)
                    Fcross2 = 0.5*(Fcross_w[jj, kk]*gammaW_plus - Fcross_u[jj, kk]*gammaU_minus)*exp_u
                    Fcross3 = 0.5*(Fcross_v[jj, kk]*gammaV_minus - Fcross_w[jj, kk]*gammaW_minus)*exp_v

                    ## Detector response summed over polarization and integrated over sky direction
                    R1[ii, 0], R1[ii, 1] = R1[ii, 0] + norm*Fplus1*rand_plus[jj, kk, ii], R1[ii, 1] + norm*Fcross1*rand_cross[jj, kk, ii]
                    R2[ii, 0], R2[ii, 1] = R2[ii, 0] + norm*Fplus2*rand_plus[jj, kk, ii], R2[ii, 1] + norm*Fcross2*rand_cross[jj, kk, ii]
                    R3[ii, 0], R3[ii, 1] = R3[ii, 0] + norm*Fplus3*rand_plus[jj, kk, ii], R3[ii, 1] + norm*Fcross3*rand_cross[jj, kk, ii] 

  

        return R1, R2, R3

#def orbiting_isgwb_mich_strain_response(object self):

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

        '''
        cdef double[:] f0 = self.f0
        cdef double[:] timearray = self.timearray
        cdef double pi_val = 3.141592653589793238462
        cdef double[:] ct = np.linspace(-1, 1, 150)
        cdef double[:] phi = np.linspace(0, 2*pi_val, 150, endpoint=False)
        cdef double[:] st = np.zeros(ct.size)
        cdef double dct = ct[1] - ct[0]
        cdef double dphi = phi[1] - phi[0]
        
        cdef double x1, x2, x3, y1, y2, y3, z1, z2, z3
        cdef double[:, :] rs1 = self.rs1
        cdef double[:, :] rs2 = self.rs2
        cdef double[:, :] rs3 = self.rs3
        cdef double[:] uvec = np.zeros(rs1[:,0].size)
        cdef double[:] vvec = np.zeros(rs1[:,0].size)
        cdef double[:] wvec = np.zeros(rs1[:,0].size)
        cdef double Lu, Lv, Lw
        cdef double x21u, y21u, z21u, x31v, y31v, z31v, x32w, y32w, z32w

        cdef int numfreqs = f0.size
        cdef int numtheta =  ct.size
        cdef int numphi =  phi.size
        cdef int numtime = timearray.size
        cdef int numdims = rs1.size
        cdef int ii, jj, kk, tt, rr, cc

        cdef double[:, :] udir = np.zeros((numphi, numtheta))
        cdef double[:, :] vdir = np.zeros((numphi, numtheta)) 
        cdef double[:, :] wdir = np.zeros((numphi, numtheta))

        cdef double[:,:] Pcontract_u = np.zeros((numphi, numtheta))
        cdef double[:,:] Pcontract_v = np.zeros((numphi, numtheta))
        cdef double[:,:] Pcontract_w = np.zeros((numphi, numtheta))

        cdef double[:,:] Ccontract_u = np.zeros((numphi, numtheta))
        cdef double[:,:] Ccontract_v = np.zeros((numphi, numtheta))
        cdef double[:,:] Ccontract_w = np.zeros((numphi, numtheta))

        cdef norm = sqrt_gsl(0.5/udir.size)

        cdef complex exp_3f0, exp_f0, exp_u, exp_v
        cdef complex exp_u_minus, exp_v_minus, exp_w_minus
        cdef complex exp_u_plus, exp_v_plus, exp_w_plus 

        # Initlize arrays for the detector reponse
        cdef complex[:,:] R1 = np.zeros((f0.size, 2), dtype='complex')
        cdef complex[:,:] R2 = np.zeros((f0.size, 2), dtype='complex')
        cdef complex[:,:] R3 = np.zeros((f0.size, 2), dtype='complex')

        cdef complex gammaU_plus,gammaV_plus,gammaW_plus,gammaU_minus,gammaV_minus,gammaW_minus
        cdef complex Fplus1,Fplus2,Fplus3,Fcross1,Fcross2,Fcross3, rand_plus, rand_cross

        for cc in range(ct.size):
            st[cc] = sqrt_gsl(1-ct[cc]**2)
        print("I'm doing the thing!")
        for tt in range(numtime):
            ## Define x/y/z for each satellite at time given by midpoints[ti]
            x1 = rs1[0][tt]
            y1 = rs1[1][tt]
            z1 = rs1[2][tt]
            x2 = rs2[0][tt]
            y2 = rs2[1][tt]
            z2 = rs2[2][tt]
            x3 = rs3[0][tt]
            y3 = rs3[1][tt]
            z3 = rs3[2][tt]
            
            ## Define vector u at time midpoints[ti]
            for rr in range(numdims):
                uvec[rr] = rs2[rr,tt] - rs1[rr,tt]
                vvec[rr] = rs3[rr,tt] - rs1[rr,tt]
                wvec[rr] = rs3[rr,tt] - rs2[rr,tt]
    
            ## Calculate arm lengths
            Lu = np.sqrt(np.dot(uvec,uvec))
            Lv = np.sqrt(np.dot(vvec,vvec))
            Lw = np.sqrt(np.dot(wvec,wvec))
            
            x21u = (x2-x1)/Lu
            y21u = (y2-y1)/Lu
            z21u = (z2-z1)/Lu
            
            x31v = (x3-x1)/Lv
            y31v = (y3-y1)/Lv
            z31v = (z3-z1)/Lv
            
            x32w = (x3-x2)/Lw
            y32w = (y3-y2)/Lw
            z32w = (z3-z2)/Lw
            
            #udir = ((x2-x1)/Lu)*np.cos(phi)*st + ((y2-y1)/Lu)*np.sin(phi)*st + ((z2-z1)/Lu)*ct
            for ii in range(numphi):
                for jj in range(numtheta):
                    udir[ii, jj] = x21u*cos_gsl(phi[ii])*st[jj] + y21u*sine_gsl(phi[ii])*st[jj] + z21u*ct[jj]
                    vdir[ii, jj] = x31v*cos_gsl(phi[ii])*st[jj] + y31v*sine_gsl(phi[ii])*st[jj] + z31v*ct[jj]
                    wdir[ii, jj] = x32w*cos_gsl(phi[ii])*st[jj] + y32w*sine_gsl(phi[ii])*st[jj] + z32w*ct[jj]
    
                    ## Calculate 1/2(u x u):eplus
                    Pcontract_u[ii, jj] = 1/2*((x21u*sine_gsl(phi[ii])-y21u*cos_gsl(phi[ii]))**2 - \
                                     (x21u*cos_gsl(phi[ii])*ct[jj]+y21u*sine_gsl(phi[ii])*ct[jj]-z21u*st[jj])**2)
                    Pcontract_v[ii, jj] = 1/2*((x31v*sine_gsl(phi[ii])-y31v*cos_gsl(phi[ii]))**2 - \
                                     (x31v*cos_gsl(phi[ii])*ct[jj]+y31v*sine_gsl(phi[ii])*ct[jj]-z31v*st[jj])**2)
                    Pcontract_w[ii, jj] = 1/2*((x32w*sine_gsl(phi[ii])-y32w*cos_gsl(phi[ii]))**2 - \
                                     (x32w*cos_gsl(phi[ii])*ct[jj]+y32w*sine_gsl(phi[ii])*ct[jj]-z32w*st[jj])**2)
                    
                    ## Calculate 1/2(u x u):ecross
                    Ccontract_u[ii, jj] = (x21u*sine_gsl(phi[ii])-y21u*cos_gsl(phi[ii])) * \
                                    (x21u*cos_gsl(phi[ii])*ct[jj]+y21u*sine_gsl(phi[ii])*ct[jj]-z21u*st[jj])
                    
                    Ccontract_v[ii, jj] = (x31v*sine_gsl(phi[ii])-y31v*cos_gsl(phi[ii])) * \
                                    (x31v*cos_gsl(phi[ii])*ct[jj]+y31v*sine_gsl(phi[ii])*ct[jj]-z31v*st[jj])
                    
                    Ccontract_w[ii, jj] = (x32w*sine_gsl(phi[ii])-y32w*cos_gsl(phi[ii])) * \
                                    (x32w*cos_gsl(phi[ii])*ct[jj]+y32w*sine_gsl(phi[ii])*ct[jj]-z32w*st[jj])
    
            # Calculate the detector response for each frequency
            for ii in range(numfreqs):
    
                exp_3f0 = cmp_exp_gsl(-3*f0[ii])
                exp_f0 = cmp_exp_gsl(-f0[ii])
                
                for jj in range(numphi):
                    for kk in range(numtheta):
    
                        exp_u_minus = cmp_exp_gsl(-f0[ii]*udir[jj, kk])
                        exp_v_minus = cmp_exp_gsl(-f0[ii]*vdir[jj, kk])
                        exp_w_minus = cmp_exp_gsl(-f0[ii]*wdir[jj, kk])
    
                        exp_u_plus, exp_v_plus, exp_w_plus   = exp_u_minus.conjugate(), exp_v_minus.conjugate(), exp_w_minus.conjugate()
    
    
                        # Calculate GW transfer function for the michelson channels
                        gammaU_plus    =    1/2 * (sinc_gsl((f0[ii])*(1 - udir[jj, kk])/pi_val)*exp_3f0 + \
                                     sinc_gsl((f0[ii])*(1 + udir[jj, kk])/pi_val)*exp_3f0) * exp_u_minus
    
    
                        gammaV_plus    =    1/2 * (sinc_gsl((f0[ii])*(1 - vdir[jj, kk])/pi_val)*exp_3f0 + \
                                     sinc_gsl((f0[ii])*(1 + vdir[jj,kk])/pi_val)*exp_3f0) * exp_v_minus
    
                        gammaW_plus    =    1/2 * (sinc_gsl((f0[ii])*(1 - wdir[jj, kk])/pi_val)*exp_3f0 + \
                                     sinc_gsl((f0[ii])*(1 + wdir[jj,kk])/pi_val)*exp_3f0) * exp_w_minus
                
                        # Calculate GW transfer function for the michelson channels
                        gammaU_minus    =    1/2 * (sinc_gsl((f0[ii])*(1 + udir[jj,kk])/pi_val)*exp_3f0 + \
                                     sinc_gsl((f0[ii])*(1 - udir[jj,kk])/pi_val)*exp_3f0) * exp_u_plus
    
                        gammaV_minus    =    1/2 * (sinc_gsl((f0[ii])*(1 + vdir[jj,kk])/pi_val)*exp_3f0 + \
                                     sinc_gsl((f0[ii])*(1 - vdir[jj,kk])/pi_val)*exp_3f0) * exp_v_plus
    
                        gammaW_minus    =    1/2 * (sinc_gsl((f0[ii])*(1 + wdir[jj,kk])/pi_val)*exp_3f0 + \
                                     sinc_gsl((f0[ii])*(1 - wdir[jj,kk])/pi_val)*exp_3f0) * exp_w_plus
            
    
                        exp_u  = cmp_exp_gsl(2*f0[ii]*udir[jj,kk])
                        exp_v  = cmp_exp_gsl(2*f0[ii]*vdir[jj,kk])
    
                        ## Michelson antenna patterns: Calculate Fplus
                        Fplus1 = Pcontract_u[jj, kk]*gammaU_plus - Pcontract_v[jj, kk]*gammaV_plus
                        Fplus2 = (Pcontract_w[jj, kk]*gammaW_plus - Pcontract_u[jj, kk]*gammaU_minus)*exp_u
                        Fplus3 = (Pcontract_v[jj, kk]*gammaV_minus - Pcontract_w[jj, kk]*gammaW_minus)*exp_v
                        ## Michelson antenna patterns: Calculate Fcross
                        Fcross1 = Ccontract_u[jj, kk]*gammaU_plus - Ccontract_v[jj, kk]*gammaV_plus
                        Fcross2 = (Ccontract_w[jj, kk]*gammaW_plus - Ccontract_u[jj, kk]*gammaU_minus)*exp_u
                        Fcross3 = (Ccontract_v[jj, kk]*gammaV_minus - Ccontract_w[jj, kk]*gammaW_minus)*exp_v
    
                        ## Directional random number
                        rand_plus  = cmp_exp_gsl(2*pi_val*unit_nrm_gsl())
                        rand_cross = cmp_exp_gsl(2*pi_val*unit_nrm_gsl())
    
                        ## Detector response summed over polarization and integrated over sky direction
                        R1[ii, 0], R1[ii, 1] = R1[ii, 0] + norm*Fplus1*rand_plus, R1[ii, 1] + norm*Fcross1*rand_cross
                        R2[ii, 0], R2[ii, 1] = R2[ii, 0] + norm*Fplus2*rand_plus, R2[ii, 1] + norm*Fcross2*rand_cross 
                        R3[ii, 0], R3[ii, 1] = R3[ii, 0] + norm*Fplus3*rand_plus, R3[ii, 1] + norm*Fcross3*rand_cross 
                        if ii==1:
                            import pdb
                            pdb.set_trace()
  

        return R1, R2, R3


'''        