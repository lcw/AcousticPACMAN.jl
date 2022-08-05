"""
2017-05-31 14:00:00
@author: Harald Ziegelwanger

The PAC-MAN model (H. Ziegelwanger and P. Reiter: "The PAC-MAN Model: Benchmark Case for Linear Acoustics in Computational Physics", Journal of Computational Physics)

This code is licensed under the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
"""


import numpy as np
import scipy.special as spsp

import pacman


def calc_mode_amplitudes(k, r_0, maxOrder, r_s, phi_s, Q=1, R_s=0, Z0Ve=0):
    """Calculates the anti-symmetric and symmetric mode amplitudes for the exterior zone.

    Calculates the anti-symmetric and symmetric mode amplitudes for the exterior zone.

    Parameters
    ----------
    k : float
        Wave number.
    r_0 : float
        Radius of the PAC-MAN.
    maxOrder : int
        Truncation order of the infinite sums.
    r_s : float
        Radius of the source position.
    phi_s : float
        Angle of the source position.
    Q : float
        Amplitude of the source.
    R_s : float
        Radius of the disk source.
    Z0Ve : float
        Amplitude of the surface velocity (multiplied with the field impedance).

    Returns
    -------
    numpy.array(dtype=complex)
        Complex amplitudes of the anti-symmetric modes in the exterior zone.
    numpy.array(dtype=complex)
        Complex amplitudes of the symmetric modes in the exterior zone.

    """

    a_asym = np.zeros([(maxOrder+1)], dtype=complex)
    a_sym = np.zeros([(maxOrder+1)], dtype=complex)

    if Q != 0:
        if r_s == np.inf:
            for n in range(0, (maxOrder+1)):
                a_asym[n] = -pacman._heavyside(n)*1j**n*spsp.jvp(n, k*r_0, 1)/spsp.h2vp(n, k*r_0, 1)*np.sin(n*phi_s)
                a_sym[n] = -pacman._heavyside(n)*1j**n*spsp.jvp(n, k*r_0, 1)/spsp.h2vp(n, k*r_0, 1)*np.cos(n*phi_s)
        else:
            if R_s == 0:
                for n in range(0, (maxOrder+1)):
                    a_asym[n] = -spsp.jvp(n, k*r_0, 1)*spsp.hankel2(n, k*r_s)/spsp.h2vp(n, k*r_0, 1)*np.sin(n*phi_s)
                    a_sym[n] = -spsp.jvp(n, k*r_0, 1)*spsp.hankel2(n, k*r_s)/spsp.h2vp(n, k*r_0, 1)*np.cos(n*phi_s)
            else:
                for n in range(0, (maxOrder+1)):
                    a_asym[n] = -pacman._heavyside(n)*(-1j*np.pi*R_s/2/k)*spsp.jvp(n, k*r_0, 1)*spsp.hankel2(n, k*r_s)*spsp.jv(1, k*R_s)/spsp.h2vp(n, k*r_0, 1)*np.sin(n*phi_s)
                    a_sym[n] = -pacman._heavyside(n)*(-1j*np.pi*R_s/2/k)*spsp.jvp(n, k*r_0, 1)*spsp.hankel2(n, k*r_s)*spsp.jv(1, k*R_s)/spsp.h2vp(n, k*r_0, 1)*np.cos(n*phi_s)

        a_asym *= Q
        a_sym *= Q

    a_sym[0] += Z0Ve/1j/spsp.h2vp(0, k*r_0, 1)

    return a_asym, a_sym


def calc_p(k, a_asym, a_sym, P):
    """Calculates the sound pressure in the exterior zone.

    Calculates the sound pressure in the exterior zone (compare Eq. 6).

    Parameters
    ----------
    k : float
        Wave number [1/m].
    a_asym : numpy.array(dtype=complex)
        Complex amplitudes of the anti-symmetric modes in the exterior zone.
    a_sym : numpy.array(dtype=complex)
        Complex amplitudes of the symmetric modes in the exterior zone.
    P : numpy.array
        Point where the sound pressure amplitude is calculated [m m m].

    Returns
    -------
    complex
        Complex sound pressure amplitude at point 'P' [Pa].

    """

    r = np.linalg.norm(P, ord=2)
    phi = np.arctan2(P[1], P[0])
    n = np.arange(0, a_sym.shape[0])

    return np.sum(a_asym*spsp.hankel2(n, k*r)*np.sin(n*phi))+np.sum(a_sym*spsp.hankel2(n, k*r)*np.cos(n*phi))
