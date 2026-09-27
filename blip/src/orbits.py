# LSS 30 June 2025 -- making a new orbits file that has definitions for keplerian orbit prescriptions as well as rigid orbits

import numpy as np
import astropy.constants as apyconst


def lisa_orbits_algebraic(tsegmid, armlength=2.5e9):
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

    times = tsegmid
    ## Semimajor axis in m
    a = 1.496e11
    ## Alpha and beta phases allow for changing of initial satellite orbital phases; default initial conditions are alphaphase=betaphase=0.
    betaphase = 0
    alphaphase = 0
    ## Orbital angle alpha(t)
    at = (2*np.pi/31557600)*times + alphaphase
    ## Eccentricity. L-dependent, so needs to be altered for time-varied arm length case.
    e = armlength/(2*a*np.sqrt(3))
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

def lisa_orbits_keplerian(tsegmid, L=2.5e9, a=1.496e11, lambda1=0, m_init1=0, kepler_order=2):


    '''
    Levi Schult 2025 01 08
    Define LISA orbital positions at the midpoint of each time integration segment
    using Keplerian orbits - arm lengths will vary.

    This is heavily-based and draws greatly from the KeplerianOrbits class
    in the LisaOrbits package: https://lisa-simulation.pages.in2p3.fr/orbits/html/latest/keplerian.html
    Orbits are the solutions to two-body problem in Newtonian gravity
    (Earth gravity is neglected). Arm flexing is minimized in next-to
    leading order in eccentricity. The math is written well at the page linked
    above.
    Parameters
    -----------
    tsegmid  :  array (N)
        A numpy array of the tsegmid for each time integration segment.

    L  :  float
        mean inter-spacecraft distance [m]. for default, give L=2.5e9.
    a  :  float
        semi-major axis for an orbital period of 1 yr [m]. Default is 1 AU=
        1.496e11 m

    lambda1  :  float
        spacecraft 1's longitude of periastron [rad] default=0
    m_init1  :  float
        spacecraft 1's mean anomaly at initial time [rad] default=0
    kepler_order  :  int
        number of iterations in the Newton-Raphson procedure. default=2
    Returns
    -----------
    rs1, rs2, rs3  :  array
        Arrays of satellite positions for each segment midpoint in timearray. e.g. rs1[1] is [x1,y1,z1] at t=midpoint[1]=timearray[1]+(segment length)/2.
    '''
    # LSS 20250108 - setting armlength to self.armlength as default
    if L is None:
        raise Exception('No armlength given!')

    times = tsegmid

    # LSS 20250108 - perturbation to tilt angle nu. This minimizes breating
    # of LISA constellation. For details, see arXiv:gr-qc/0507105
    delta = 5.0 / 8.0

    # LSS 20250108 - orbital parameter used for series expansions
    alpha = L / (2 * a)

    # LSS 20250108 - LISA constellation's  orbital tilt angle to the ecliptic
    nu = (np.pi / 3.0) + (delta * alpha)

    # LSS 20250108 - Orbital eccentricity
    e = np.sqrt(1 + ((4 * alpha * np.cos(nu)) / np.sqrt(3)) + ((4 * alpha**2) / 3)) - 1

    ## Alpha and beta phases allow for changing of initial satellite orbital phases; default initial conditions are alphaphase=betaphase=0.
    betaphase = 0
    alphaphase = 0

    # LSS 20250108 - now we calculate things that are necessary later for
    # position calculations
    tan_i = alpha * np.sin(nu) / ((np.sqrt(3) / 2) + alpha * np.cos(nu))
    cos_i = 1 / np.sqrt(1 + tan_i**2)
    sin_i = tan_i * cos_i
    n = np.sqrt(apyconst.GM_sun.value / a**3)

    ## Initialize arrays
    # LSS 20250109 - beta_n is self.theta - betaphase in lisa orbits.
    # I think that m_init1 in lisa orbits is betaphase in blip
    # and lambda1 is alphaphase in blip.
    # so m_init = betaphase - beta_n
    # alpha_k is iterating over alphaphase for different sc.
    # choosing to follow blip convention: theta+betaphase
    # rather than betaphase - theta in lisa_orb
    beta_n = (2/3)*np.pi*np.array([0,1,2])+betaphase # (3,) or (M,)
    alpha_k = beta_n + alphaphase # (3,)
    sin_alpha = np.sin(alpha_k) # (3,)
    cos_alpha = np.cos(alpha_k) # (3,)

    gr_const = ((n * a) / apyconst.c.value)**2

    ## Orbital angle alpha(t)
    #at = (2*np.pi/31557600)*times + alphaphase

    r'''
    Estimate the eccentric anomaly.
    This is heavily-based and draws greatly from the KeplerianOrbits class
    in the LisaOrbits package: https://lisa-simulation.pages.in2p3.fr/orbits/html/latest/keplerian.html

    Their docstring explains the math:
    This uses an iterative Newton-Raphson method to solve the Kepler equation,
    starting from a low eccentricity expansion of the solution.
    .. math::
        \psi_k - e \sin \psi_k = m_k(t) \qc
    with :math:`m_k(t)` the mean anomaly.
    We use ``kepler_order`` iterations. For low eccentricity, the convergence rate
    of this iterative scheme is of the order of :math:`e^2`. Typically for LISA
    spacecraft (characterized by a small eccentricity 0.005), the iterative
    procedure converges in one iteration using double precision.
    '''

    sc_index = np.array([0, 1, 2])
    since_init_time = lambda t_array : t_array - t_array[0]
    # LSS 20250110 - this makes a (N, M) array where N is size of times and M is num spacecraft
    m = beta_n[np.newaxis] + n * since_init_time(times)[np.newaxis].T


    ecc_anomaly = np.array((tsegmid.shape[0], 3))

    # The following expression is valid up to e**4
    ecc_anomaly = m + (e - e**3/8) * np.sin(m) + 0.5 * e**2 * np.sin(2 * m) \
        + 3/8 * e**3 * np.sin(3 * m) # (N, M)
    # Standard Newton-Raphson iterative procedure
    for _ in range(kepler_order):
        error = ecc_anomaly - e * np.sin(ecc_anomaly) - m # (N, M)
        ecc_anomaly -= error / (1 - e * np.cos(ecc_anomaly)) # (N, M)

    # Compute eccentric anomaly
    psi = ecc_anomaly # (N, M)
    cos_psi = np.cos(psi) # (N, M)
    sin_psi = np.sin(psi) # (N, M)

    # Reference position
    ref_x = a * cos_i * (cos_psi - e) # (N, M)
    ref_y = a * np.sqrt(1 - e**2) * sin_psi # (N, M)
    ref_z = -a * sin_i * (cos_psi - e) # (N, M)
    # Spacecraft position
    sc_x = cos_alpha[np.newaxis, sc_index] * ref_x \
        - sin_alpha[np.newaxis, sc_index] * ref_y # (N, M)
    sc_y = sin_alpha[np.newaxis, sc_index] * ref_x \
        + cos_alpha[np.newaxis, sc_index] * ref_y # (N, M)
    sc_z = ref_z # (N, M)

    ## Construct position vectors r_n
    rs1 = np.array([sc_x[:, 0],sc_y[:, 0],sc_z[:, 0]])
    rs2 = np.array([sc_x[:, 1],sc_y[:, 1],sc_z[:, 1]])
    rs3 = np.array([sc_x[:, 2],sc_y[:, 2],sc_z[:, 2]])

    return rs1, rs2, rs3
