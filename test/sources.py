"""
2017-05-31 14:00:00
@author: Harald Ziegelwanger

The PAC-MAN model (H. Ziegelwanger and P. Reiter: "The PAC-MAN Model: Benchmark Case for Linear Acoustics in Computational Physics", Journal of Computational Physics)

This code is licensed under the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
"""


import numpy as np
import scipy.special as spsp


def line_p(k, r_s, phi_s, P):
    """Calculates the sound pressure of a line source.

    Calculates the sound pressure at point 'P' for an incoming wave originating from a line source.

    Parameters
    ----------
    k : float
        Wave number [1/m].
    r_s : float
        Radius of the source position [m].
    phi_s : float
        Angle of the source position [rad].
    P : numpy.array
        Point where the sound pressure amplitude is calculated [m m m].

    Returns
    -------
    complex
        Complex sound pressure amplitude at point 'P' [Pa].

    """

    x = r_s*np.cos(phi_s)
    y = r_s*np.sin(phi_s)

    x1 = P[0]-x
    y1 = P[1]-y

    r = np.linalg.norm(np.array([x1, y1]), ord=2)

    return spsp.hankel2(0, k*r)


def disk_p(k, r_s, phi_s, R_s, P):
    """Calculates the sound pressure of a disk source.

    Calculates the sound pressure at point 'P' for an incoming wave originating from a disk source.

    Parameters
    ----------
    k : float
        Wave number [1/m].
    r_s : float
        Radius of the source position [m].
    phi_s : float
        Angle of the source position [rad].
    R_s : float
        Radius of the disk source [m].
    P : numpy.array
        Point where the sound pressure amplitude is calculated [m m m].

    Returns
    -------
    complex
        Complex sound pressure amplitude at point 'P' [Pa].

    """

    x = r_s*np.cos(phi_s)
    y = r_s*np.sin(phi_s)

    x1 = P[0]-x
    y1 = P[1]-y

    r = np.linalg.norm(np.array([x1, y1]), ord=2)

    if r < R_s:
        return (1-1j*np.pi/2)/k*R_s*spsp.hankel2(1, k*R_s)*spsp.jv(0, k*r)
    else:
        return -1j*np.pi/2/k*R_s*spsp.hankel2(0, k*r)*spsp.jv(1, k*R_s)


def plane_p(k, phi_s, P):
    """Calculates the sound pressure of a plane wave.

    Calculates the sound pressure at point 'P' for an incoming plane wave.

    Parameters
    ----------
    k : float
        Wave number [1/m].
    phi_s : float
        Angle of the source position [rad].
    P : numpy.array
        Point where the sound pressure amplitude is calculated [m m m].

    Returns
    -------
    complex
        Complex sound pressure amplitude at point 'P' [Pa].

    """

    r = np.linalg.norm(P, ord=2)
    phi = np.arctan2(P[1], P[0])

    return np.cos(k*r*np.cos(phi-phi_s))+1j*np.sin(k*r*np.cos(phi-phi_s))
