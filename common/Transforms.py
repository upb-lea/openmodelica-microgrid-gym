"""
Created on Fri Jan 24 16:19:01 2020

@author: jarren

Common static transforms used commonly in voltage/power inverter control systems
"""

import numpy as np


def dq0_abc(dq0: np.ndarray, theta: float) -> np.ndarray:
    """
    Transforms from DQ frame to the abc frame using the provided angle theta

    :param dq0: The values in the dq0 reference frame
    :param theta: The angle to transform [In Radians]

    :return abc: The transformed space in the abc frame
    """
    theta_rad = theta
    a = (np.cos(theta_rad) * dq0[0] -
         np.sin(theta_rad) * dq0[1] +
         dq0[2])
    theta2 = theta_rad - 2 * np.pi / 3
    b = (np.cos(theta2) * dq0[0] -
         np.sin(theta2) * dq0[1] +
         dq0[2])
    theta2 = theta_rad + 2 * np.pi / 3
    c = (np.cos(theta2) * dq0[0] -
         np.sin(theta2) * dq0[1] +
         dq0[2])

    return np.array([a, b, c])


def dq0_to_abc_cos_sin(dq0: np.ndarray, cos: float, sin: float) -> np.ndarray:
    """
    Transforms from DQ frame to the abc frame using the provided cos-sin
    This implementation tries to improve on the dq0Toabc transform by
    utilising the provided cos-sin to minimise calls to calculate the
    cossine etc

    :param dq0: The values in the dq0 reference frame
    :param cossine: The cossine of the angle [cos(theta), sine (theta)]

    :return abc: The transformed space in the abc frame
    """
    abc = np.empty(3)
    abc[0] = np.array([cos, -sin, 1]) @ dq0

    # implements the cos(a-2pi/3) using cos (A+B) expansion etc
    for i in range(1, 3):
        cos_shift = -cos * 0.5 - sin * 0.866 * (-1) ** i
        sin_shift = -sin * 0.5 + cos * 0.866 * (-1) ** i
        abc[i] = np.array([cos_shift, -sin_shift, 1]) @ dq0
    return abc


def dq0_to_abc_cos_sin_power_inv(dq0: np.ndarray, cos: float, sin: float) -> np.ndarray:
    """
    Transforms from DQ frame to the abc frame using the provided cos-sin
    This implementation tries to improve on the dq0Toabc transform by
    utilising the provided cos-sin to minimise calls to calculate the
    cossine etc.

    Provides the Power Invariant transform (multiplied by
    SQRT(3/2) = 1.224744871391589)

    :param dq0: The values in the dq0 reference frame
    :param cossine: The cossine of the angle [cos(theta), sine (theta)]

    :return abc: The transformed space in the abc frame
    """
    return dq0_to_abc_cos_sin(dq0, cos, sin) * 1.224744871391589


def abc_to_dq0(abc, theta):
    """
    Transforms from abc frame to the dq0 frame using the provided angle

    :param abc: The values in the abc reference frame
    :param theta: The angle [radians]

    :return dq0: The transformed space in the abc frame
    """
    return abc_to_dq0_cos_sin(abc, *cos_sin(theta))


def abc_to_dq0_cos_sin(abc, cos, sin):
    """
    Transforms from abc frame to the dq0 frame using the provided cos-sin
    This implementation tries to improve by utilising the provided cos-sin
    to minimise calls to calculate the cossine etc

    :param abc: The values in the abc reference frame
    :param cossine: The cossine of the angle [cos(theta), sine (theta)]

    :return dq0: The transformed space in the abc frame
    """
    # implements the cos(a-2pi/3) using cos (A+B) expansion etc
    cos_shift_neg = cos * (-0.5) - sin * (-0.866)
    sin_shift_neg = sin * (-0.5) + cos * (-0.866)
    # implements the cos(a+2pi/3) using cos (A+B) expansion etc
    cos_shift_pos = cos * (-0.5) - sin * (0.866)
    sin_shift_pos = sin * (-0.5) + cos * (0.866)

    # Calculation for d-axis aligned with A axis
    d = (2 / 3) * (cos * abc[0] +
                   cos_shift_neg * abc[1] +
                   cos_shift_pos * abc[2])
    q = (2 / 3) * (-sin * abc[0] -
                   sin_shift_neg * abc[1] -
                   sin_shift_pos * abc[2])

    z = (1 / 3) * (abc[0] + abc[1] + abc[2])

    return np.array([d, q, z])


def abctoAlphaBeta(abc) -> np.ndarray:
    """
    Transforms from abc frame to the alpha-beta frame
    :param abc: The values in the abc reference frame
    :return [alpha,beta]: The transformed alpha beta results
    """

    alpha = (2 / 3) * (abc[0] - 0.5 * abc[1] - 0.5 * abc[2])
    beta = (2 / 3) * (0.866 * abc[1] - 0.866 * abc[2])

    return np.array([alpha, beta])


def cos_sin(theta) -> tuple:
    """
    Transforms from provided angle to the relavent cossine values
    :param theta: The angle [In RADIANS]
    :return [alpha,beta]: The resultant cossine
    """
    return np.cos(theta), np.sin(theta)


def inst_rms(arr: np.ndarray):
    return np.linalg.norm(arr) / 1.732050807568877


def inst_power(varr: np.ndarray, iarr: np.ndarray):
    # scalar product
    return varr @ iarr


def inst_reactive(varr: np.ndarray, iarr: np.ndarray):
    vline = np.array([varr[1] - varr[2], varr[2], varr[0]])
    return -0.5773502691896258 * vline @ iarr
