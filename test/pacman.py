"""
2017-05-31 14:00:00
@author: Harald Ziegelwanger

The PAC-MAN model (H. Ziegelwanger and P. Reiter: "The PAC-MAN Model: Benchmark Case for Linear Acoustics in Computational Physics", Journal of Computational Physics)

This code is licensed under the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
"""


import numpy as np
import scipy.special as spsp

import sources


def _kronecker(x, y):
    if x == y:
        return 1
    else:
        return 0


def _heavyside(x):
    if x == 0:
        return 1
    else:
        return 2


def _I_cos(x, y, phi_0):
    if x != y:
        return (np.sin((x+y)*phi_0)/(x+y)/phi_0+np.sin((x-y)*phi_0)/(x-y)/phi_0)/2
    else:
        if x == 0:
            return 1
        else:
            return 0.5*(1+np.sin(2*x*phi_0)/2/x/phi_0)


def _I_sin(x, y, phi_0):
    if x != y:
        return (np.sin((x-y)*phi_0)/(x-y)/phi_0-np.sin((x+y)*phi_0)/(x+y)/phi_0)/2
    else:
        if x == 0:
            return 0
        else:
            return 0.5*(1-np.sin(2*x*phi_0)/2/x/phi_0)


def _P_inc(n_s, k, r, N, r_s, phi_s, R_s, Q, asym):
    if Q == 0:
        return 0
    else:
        if r_s == np.inf:  # plane wave
            P_inc = _heavyside(n_s)*(1j)**n_s*spsp.jv(n_s, k*r)
        elif R_s == 0:  # line source
            if r_s > r:  # exterior zone
                P_inc = _heavyside(n_s)*spsp.jv(n_s, k*r)*spsp.hankel2(n_s, k*r_s)
            else:  # interior zone
                if asym:
                    if (np.sin(n_s*phi_s) == 0) | (n_s > int((np.sqrt(np.ceil(k*r_s))*4-0.5)*N)):
                        P_inc = 0
                    else:
                        P_inc = -_interior_line_source_modes(n_s, k, r, N, r_s, phi_s, asym)*spsp.hankel2(n_s, k*r)/np.sin(n_s*phi_s)
                else:
                    if (np.cos(n_s*phi_s) == 0) | (n_s > int((np.sqrt(np.ceil(k*r_s))*4-1)*N)):
                        P_inc = 0
                    else:
                        P_inc = -_interior_line_source_modes(n_s, k, r, N, r_s, phi_s, asym)*spsp.hankel2(n_s, k*r)/np.cos(n_s*phi_s)
        else:  # disc source
            P_inc = _heavyside(n_s)*spsp.jv(n_s, k*r)*spsp.hankel2(n_s, k*r_s)*spsp.jv(1, k*R_s)*(-1j)*np.pi*R_s/2/k

        return Q*P_inc


def _V_inc(i, k, r, N, r_s, phi_s, R_s, Q, asym):
    if Q == 0:
        return 0
    else:
        if r_s == np.inf:  # plane wave
            V_inc = _heavyside(i)*(1j)**i*spsp.jvp(i, k*r, 1)
        elif R_s == 0:  # line source
            if r_s > r:  # exterior zone
                V_inc = _heavyside(i)*spsp.jvp(i, k*r, 1)*spsp.hankel2(i, k*r_s)
            else:  # interior zone
                V_inc = 0
                if asym:
                    if np.sin(i*phi_s) != 0:
                        for n_s in range(0, int(np.sqrt(np.ceil(k*r_s))*4)):
                            V_inc -= _interior_line_source_modes((n_s+0.5)*N, k, r, N, r_s, phi_s, asym)*spsp.h2vp((n_s+0.5)*N, k*r, 1)*_I_sin((n_s+0.5)*N, i, np.pi/N)
                        V_inc *= 2/np.sin(i*phi_s)/N
                else:
                    if np.cos(i*phi_s) != 0:
                        for n_s in range(0, int(np.sqrt(np.ceil(k*r_s))*4)):
                            V_inc -= _interior_line_source_modes(n_s*N, k, r, N, r_s, phi_s, asym)*spsp.h2vp(n_s*N, k*r, 1)*_I_cos(n_s*N, i, np.pi/N)
                        V_inc *= _heavyside(i)/np.cos(i*phi_s)/N
        else:  # disc source
            V_inc = _heavyside(i)*spsp.jvp(i, k*r, 1)*spsp.hankel2(i, k*r_s)*spsp.jv(1, k*R_s)*(-1j)*np.pi*R_s/2/k

        return Q*V_inc


def _static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@_static_vars(c_asym=np.array([]), c_sym=np.array([]), N=0)
def _interior_line_source_modes(n_s, k, r_0, N, r_s, phi_s, asym):
    if (_interior_line_source_modes.c_asym.size == 0) | (_interior_line_source_modes.c_sym.size == 0) | (_interior_line_source_modes.N != N):
        phi_0 = np.pi/N
        M_s = int(np.sqrt(np.ceil(k*r_s))*4)

        f_phi = np.zeros([2*M_s+1], dtype=complex)
        phi = np.zeros([2*M_s+1])
        for i in range(0, 2*M_s+1):
            phi[i] = -phi_0+2*phi_0/(2*M_s)*i
            for m in range(0, N):
                f_phi[i] += sources.line_p(k, r_s, 2*phi_0*m+(-1)**m*phi_s, r_0*np.array([np.cos(phi[i]), np.sin(phi[i]), 0]))
        f_phi_even = (f_phi+f_phi[::-1])/2
        f_phi_odd = (f_phi-f_phi[::-1])/2

        A_asym = np.zeros([M_s, M_s], dtype=complex)
        A_sym = np.zeros([M_s, M_s], dtype=complex)
        b_asym = np.zeros([M_s], dtype=complex)
        b_sym = np.zeros([M_s], dtype=complex)
        b_asym = f_phi_odd[M_s+1::]
        b_sym = f_phi_even[M_s+1::]
        phi_tmp = phi[M_s+1::]
        for i in range(0, M_s):
            for m_s in range(0, M_s):
                A_asym[i, m_s] = spsp.hankel2((m_s+0.5)*N, k*r_0)*np.sin((m_s+0.5)*N*phi_tmp[i])
                A_sym[i, m_s] = spsp.hankel2(m_s*N, k*r_0)*np.cos(m_s*N*phi_tmp[i])
        _interior_line_source_modes.c_asym = np.linalg.solve(A_asym, b_asym)
        _interior_line_source_modes.c_sym = np.linalg.solve(A_sym, b_sym)
        _interior_line_source_modes.N = N

    if ((n_s % N) == 0.5*N) & (asym is True):
        return _interior_line_source_modes.c_asym[int(n_s/N-0.5)]
    elif ((n_s % N) == 0) & (asym is False):
        return _interior_line_source_modes.c_sym[int(n_s/N)]
    else:
        return 0


def build_matrix(k, r_0, N, maxOrder):
    """Builds the system matrices.

    Builds the system matrices defined in Eqs. 54 and 55.

    Parameters
    ----------
    k : float
        Wave number [1/m].
    r_0 : float
        Radius of the PAC-MAN [m].
    N : int
        The PAC-MAN constant.
    maxOrder : int
        Truncation order of the infinite sums.

    Returns
    -------
    numpy.array(dtype=complex)
        System matrix of the equation system for the anti-symmetric modes.
    numpy.array(dtype=complex)
        System matrix of the equation system for the symmetric modes.

    """

    phi_0 = np.pi/N

    A_asym = np.zeros([(maxOrder+1), (maxOrder+1)], dtype=complex)
    A_sym = np.zeros([(maxOrder+1), (maxOrder+1)], dtype=complex)
    for i in range(0, (maxOrder+1)):
        for n in range(0, maxOrder+1):
            for eta in range(0, int(np.floor((maxOrder+1)/N))):
                A_asym[i, n] -= 2*spsp.jvp((eta+0.5)*N, k*r_0, 1)/spsp.jv((eta+0.5)*N, k*r_0)*_I_sin(n, (eta+0.5)*N, phi_0)*_I_sin(i, (eta+0.5)*N, phi_0)
                A_sym[i, n] -= _heavyside(eta*N)*spsp.jvp(eta*N, k*r_0, 1)/spsp.jv(eta*N, k*r_0)*_I_cos(n, eta*N, phi_0)*_I_cos(i, eta*N, phi_0)
            A_asym[i, n] *= spsp.hankel2(n, k*r_0)/N
            A_sym[i, n] *= spsp.hankel2(n, k*r_0)/N
            A_asym[i, n] += (_kronecker(n, i)-_kronecker(i, 0))/2*spsp.h2vp(i, k*r_0, 1)
            A_sym[i, n] += _kronecker(n, i)/_heavyside(i)*spsp.h2vp(i, k*r_0, 1)

    return A_asym, A_sym


def build_rhs(k, r_0, N, maxOrder, r_s, phi_s, Q=1, R_s=0, Z0Ve=0):
    """Builds the right-hand sides.

    Builds the right-hand sides defined in Eqs. 54 and 55.

    Parameters
    ----------
    k : float
        Wave number [1/m].
    r_0 : float
        Radius of the PAC-MAN [m].
    N : int
        The PAC-MAN constant.
    maxOrder : int
        Truncation order of the infinite sums.
    r_s : float
        Radius of the source position [m].
    phi_s : float
        Angle of the source position [rad].
    Q : float
        Amplitude of the source.
    R_s : float
        Radius of the disk source [m].
    Z0Ve : float
        Amplitude of the surface velocity (multiplied by the field impedance).

    Returns
    -------
    numpy.array(dtype=complex)
        right-hand side of the equation system for the anti-symmetric modes.
    numpy.array(dtype=complex)
        Right-hand side of the equation system for the symmetric modes.

    """

    phi_0 = np.pi/N

    b_asym = np.zeros([(maxOrder+1)], dtype=complex)
    b_sym = np.zeros([(maxOrder+1)], dtype=complex)

    for i in range(0, (maxOrder+1)):
        for eta in range(0, int(np.floor((maxOrder+1)/N))):
            P_inc_asym = 0
            P_inc_sym = 0
            for n_s in range(0, maxOrder+1):
                P_inc_asym += _P_inc(n_s, k, r_0, N, r_s, phi_s, R_s, Q, True)*np.sin(n_s*phi_s)*_I_sin(n_s, (eta+0.5)*N, phi_0)
                P_inc_sym += _P_inc(n_s, k, r_0, N, r_s, phi_s, R_s, Q, False)*np.cos(n_s*phi_s)*_I_cos(n_s, eta*N, phi_0)
            b_asym[i] += 2*spsp.jvp((eta+0.5)*N, k*r_0, 1)/spsp.jv((eta+0.5)*N, k*r_0)*_I_sin(i, (eta+0.5)*N, phi_0)*P_inc_asym
            b_sym[i] += _heavyside(eta*N)*spsp.jvp(eta*N, k*r_0, 1)/spsp.jv(eta*N, k*r_0)*_I_cos(i, eta*N, phi_0)*P_inc_sym
        b_asym[i] /= N
        b_sym[i] /= N
        b_asym[i] -= 1/2*_V_inc(i, k, r_0, N, r_s, phi_s, R_s, Q, True)*np.sin(i*phi_s)
        b_sym[i] -= 1/_heavyside(i)*_V_inc(i, k, r_0, N, r_s, phi_s, R_s, Q, False)*np.cos(i*phi_s)
        b_sym[i] += (_I_cos(i, 0, phi_0)/N-_kronecker(0, i)/_heavyside(i))*Z0Ve*1j

    return b_asym, b_sym


def calc_mode_amplitudes(k, r_0, N, maxOrder, r_s, phi_s, Q=1, R_s=0, Z0Ve=0):
    """Calculates the anti-symmetric and symmetric mode amplitudes for the exterior zone.

    Calculates the anti-symmetric and symmetric mode amplitudes for the exterior zone (compare Eqs. 6 and 7) by solving the linear systems of equations defined in Eqs. 54 and 55.

    Parameters
    ----------
    k : float
        Wave number [1/m].
    r_0 : float
        Radius of the PAC-MAN [m].
    N : int
        The PAC-MAN constant.
    maxOrder : int
        Truncation order of the infinite sums.
    r_s : float
        Radius of the source position [m].
    phi_s : float
        Angle of the source position [rad].
    Q : float
        Amplitude of the source.
    R_s : float
        Radius of the disk source [m].
    Z0Ve : float
        Amplitude of the surface velocity (multiplied with the field impedance).

    Returns
    -------
    numpy.array(dtype=complex)
        Complex amplitudes of the anti-symmetric modes in the exterior zone.
    numpy.array(dtype=complex)
        Complex amplitudes of the symmetric modes in the exterior zone.

    """

    A_asym, A_sym = build_matrix(k, r_0, N, maxOrder)
    b_asym, b_sym = build_rhs(k, r_0, N, maxOrder, r_s, phi_s, Q, R_s, Z0Ve)
    a_asym = np.linalg.solve(A_asym[1:, 1:], b_asym[1:])
    a_sym = np.linalg.solve(A_sym, b_sym)

    return np.append(0, a_asym), a_sym


def calc_b(a_asym, a_sym, k, r_0, N, maxOrder, r_s, phi_s, R_s=0, Q=1):
    """Calculates the anti-symmetric and symmetric mode amplitudes for the interior zone.

    Calculates the anti-symmetric and symmetric mode amplitudes for the interior zone (compare Eqs. 52 and 53).

    Parameters
    ----------
    a_asym : numpy.array(dtype=complex)
        Complex amplitudes of the anti-symmetric modes in the exterior zone.
    a_sym : numpy.array(dtype=complex)
        Complex amplitudes of the symmetric modes in the exterior zone.
    k : float
        Wave number [1/m].
    r_0 : float
        Radius of the PAC-MAN [m].
    N : int
        The PAC-MAN constant.
    maxOrder : int
        Truncation order of the infinite sums.
    r_s : float
        Radius of the source position [m].
    phi_s : float
        Angle of the source position [rad].
    R_s : float
        Radius of the disk source [m].
    Q : float
        Amplitude of the source.

    Returns
    -------
    numpy.array(dtype=complex)
        Complex amplitudes of the anti-symmetric modes in the interior zone.
    numpy.array(dtype=complex)
        Complex amplitudes of the symmetric modes in the interior zone.

    """

    phi_0 = np.pi/N
    b_asym = np.zeros_like(a_asym, dtype=complex)
    b_sym = np.zeros_like(a_sym, dtype=complex)

    for eta in range(0, maxOrder+1):
        for n in range(0, maxOrder+1):
            b_asym[eta] += a_asym[n]*spsp.hankel2(n, k*r_0)*_I_sin(n, (eta+0.5)*N, phi_0)
            b_sym[eta] += a_sym[n]*spsp.hankel2(n, k*r_0)*_I_cos(n, eta*N, phi_0)
        for n_s in range(0, maxOrder+1):
            b_asym[eta] += _P_inc(n_s, k, r_0, N, r_s, phi_s, R_s, Q, True)*np.sin(n_s*phi_s)*_I_sin(n_s, (eta+0.5)*N, phi_0)
            b_sym[eta] += _P_inc(n_s, k, r_0, N, r_s, phi_s, R_s, Q, False)*np.cos(n_s*phi_s)*_I_cos(n_s, eta*N, phi_0)
        b_asym[eta] *= 2/spsp.jv((eta+0.5)*N, k*r_0)
        b_sym[eta] *= _heavyside(eta*N)/spsp.jv(eta*N, k*r_0)

    return b_asym, b_sym


def calc_p_interior_zone(k, b_asym, b_sym, P):
    """Calculates the sound pressure in the interior zone.

    Calculates the sound pressure in the interior zone (compare Eq. 13).

    Parameters
    ----------
    k : float
        Wave number [1/m].
    b_asym : numpy.array(dtype=complex)
        Complex amplitudes of the anti-symmetric modes in the interior zone.
    b_sym : numpy.array(dtype=complex)
        Complex amplitudes of the symmetric modes in the interior zone.
    P : numpy.array
        Point where the sound pressure amplitude is calculated [m m m].

    Returns
    -------
    complex
        Complex sound pressure amplitude at point 'P' [Pa].

    """

    r = np.linalg.norm(P, ord=2)
    phi = np.arctan2(P[1], P[0])
    n = np.arange(0, b_sym.shape[0])

    return np.sum(b_asym*spsp.jv(n, k*r)*np.sin(n*phi))+np.sum(b_sym*spsp.jv(n, k*r)*np.cos(n*phi))
