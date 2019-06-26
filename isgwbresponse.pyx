'''
A Cython module to perform efficient calculation of the orbiting LISA detector response.

Calcualte the Antenna pattern/ detector transfer function functions to an isotropic SGWB for an equal-arm orbiting LISA using A, E and T TDI channels. Note that since this is the response to an isotropic background, the response function is integrated over sky direction and averaged over polarization. The angular integral is a linear and rectangular in the cos(theta) and phi space.  Note that f0 is (pi*L*f)/c and is input as an array. The detector response is calculated based on LISA's position at the midpoint of each time segment. 

The resulting response arrays are saved to three .txt files in the current directory for reuse.

Parameters
-----------

f0   : float
    A numpy array of scaled frequencies (see above for def)

rs1, rs2, rs3  :  array
    Satellite position vectors.
    
ti  :  float
    timearray index    

Returns
---------

R1, R2 and R3   :   float
    Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarization.
'''
from __future__ import division
from __future__ cimport division
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

CDTYPE = np.complex128
ctypedef np.complex128_t CDTYPE_t

def cython_tdi_isgwb_response(np.ndarray[DTYPE_t, ndim=1] f0, np.ndarray[DTYPE_t, ndim=1] midpoints, np.ndarray[DTYPE_t, ndim=2] rs1, np.ndarray[DTYPE_t, ndim=2] rs2, np.ndarray[DTYPE_t, ndim=2] rs3): 
    
    cdef np.ndarray[long int, ndim=1] timeindices = np.arange(len(midpoints))
    
    cdef np.ndarray[DTYPE_t, ndim=1] tt = np.arange(-1, 1, 0.01)
    cdef np.ndarray[DTYPE_t, ndim=1] pp = np.arange(0, 2*np.pi, np.pi/100)

    cdef np.ndarray[DTYPE_t, ndim=2] ct, phi
    
    [ct, phi] = np.meshgrid(tt,pp)
    cdef DTYPE_t dct = ct[0, 1] - ct[0,0]
    cdef DTYPE_t dphi = phi[1,0] - phi[0,0]
    cdef np.ndarray[DTYPE_t, ndim=2] st = np.sqrt(1-ct**2)
    
    # Initlize arrays for the detector reponse
    cdef np.ndarray[DTYPE_t, ndim=2] R1 = np.zeros((len(timeindices),f0.size))
    cdef np.ndarray[DTYPE_t, ndim=2] R2 = np.zeros((len(timeindices),f0.size))
    cdef np.ndarray[DTYPE_t, ndim=2] R3 = np.zeros((len(timeindices),f0.size))
    
    ## Type remaining variables
    cdef long int ti, ii
    cdef DTYPE_t x1, y1, z1, x2, y2, z2, x3, y3, z3, Lu, Lv, Lw
    cdef np.ndarray[DTYPE_t, ndim=1] uvec, vvec, wvec
    cdef np.ndarray[DTYPE_t, ndim=2] udir, vdir, wdir
    cdef np.ndarray[CDTYPE_t, ndim=2] gammaU, gammaV, gammaW, Fplus_u, Fplus_v, Fplus_w, Fcross_u, Fcross_v, Fcross_w
    cdef np.ndarray[CDTYPE_t, ndim=2] Fplus1, Fplus2, Fplus3, Fcross1, Fcross2, Fcross3, FAplus, FEplus, FTplus, FAcross, FEcross, FTcross
    
    for ti in timeindices:
        ## Define x/y/z for each satellite at time given by timearray[ti]
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

            Fplus_u   = 1/2*((((x2-x1)/Lu)*np.sin(phi)-((y2-y1)/Lu)*np.cos(phi))**2 - \
                         (((x2-x1)/Lu)*np.cos(phi)*ct+((y2-y1)/Lu)*np.sin(phi)*ct-((z2-z1)/Lu)*st)**2)*gammaU

            Fplus_v   = 1/2*((((x3-x1)/Lv)*np.sin(phi)-((y3-y1)/Lv)*np.cos(phi))**2 - \
                         (((x3-x1)/Lv)*np.cos(phi)*ct+((y3-y1)/Lv)*np.sin(phi)*ct-((z3-z1)/Lv)*st)**2)*gammaV

            Fplus_w   = 1/2*((((x3-x2)/Lw)*np.sin(phi)-((y3-y2)/Lw)*np.cos(phi))**2 - \
                         (((x3-x2)/Lw)*np.cos(phi)*ct+((y3-y2)/Lw)*np.sin(phi)*ct-((z3-z2)/Lw)*st)**2)*gammaW


            ## Michelson Channel Antenna patterns for x pol
            ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross
            Fcross_u  = (((x2-x1)/Lu)*np.sin(phi)-((y2-y1)/Lu)*np.cos(phi)) * (((x2-x1)/Lu)*np.cos(phi)*ct+((y2-y1)/Lu)*np.sin(phi)*ct-((z2-z1)/Lu)*st) * gammaU
            Fcross_v  = (((x3-x1)/Lv)*np.sin(phi)-((y3-y1)/Lv)*np.cos(phi)) * (((x3-x1)/Lv)*np.cos(phi)*ct+((y3-y1)/Lv)*np.sin(phi)*ct-((z3-z1)/Lv)*st) * gammaV
            Fcross_w  = (((x3-x2)/Lw)*np.sin(phi)-((x3-x2)/Lw)*np.cos(phi)) * (((x3-x2)/Lw)*np.cos(phi)*ct+((y3-y2)/Lw)*np.sin(phi)*ct-((z3-z2)/Lw)*st) * gammaW


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
            R1[ti][ii] = dct*dphi/(4*np.pi)*np.sum((np.absolute(FAplus))**2 + (np.absolute(FAcross))**2)
            R2[ti][ii] = dct*dphi/(4*np.pi)*np.sum((np.absolute(FEplus))**2 + (np.absolute(FEcross))**2)
            R3[ti][ii] = dct*dphi/(4*np.pi)*np.sum((np.absolute(FTplus))**2 + (np.absolute(FTcross))**2)
        


    np.savetxt('R1array.txt',R1)
    np.savetxt('R2array.txt',R2)
    np.savetxt('R3array.txt',R3)
    
    return R1, R2, R3