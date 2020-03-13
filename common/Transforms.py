# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:19:01 2020

@author: jarren

Common static transforms used commonly in voltage/power inverter control systems
"""

import math


def dq0Toabc(dq0, theta):
    """
    Transforms from DQ frame to the abc frame using the provided angle theta

    :param dq0: The values in the dq0 reference frame
    :param theta: The angle to transform [In Radians]

    :return abc: The transformed space in the abc frame
    """
    theta_rad = theta
    a = (math.cos(theta_rad) * dq0[0] -
         math.sin(theta_rad) * dq0[1] +
         dq0[2])
    theta2 = theta_rad - 2 * math.pi / 3
    b = (math.cos(theta2) * dq0[0] -
         math.sin(theta2) * dq0[1] +
         dq0[2])
    theta2 = theta_rad + 2 * math.pi / 3
    c = (math.cos(theta2) * dq0[0] -
         math.sin(theta2) * dq0[1] +
         dq0[2])

    return [a, b, c]


def dq0_to_abc_cos_sin(dq0, cossin):
    """
    Transforms from DQ frame to the abc frame using the provided cos-sin
    This implementation tries to improve on the dq0Toabc transform by
    utilising the provided cos-sin to minimise calls to calculate the
    cossine etc

    :param dq0: The values in the dq0 reference frame
    :param cossine: The cossine of the angle [cos(theta), sine (theta)]

    :return abc: The transformed space in the abc frame
    """
    cos = cossin[0]
    sin = cossin[1]
    a = (cossin[0] * dq0[0] -
         cossin[1] * dq0[1] +
         dq0[2])
    # implements the cos(a-2pi/3) using cos (A+B) expansion etc
    cos_shift = cos * (-0.5) - sin * (-0.866)
    sin_shift = sin * (-0.5) + cos * (-0.866)
    b = (cos_shift * dq0[0] -
         sin_shift * dq0[1] +
         dq0[2])

    cos_shift = cos * (-0.5) - sin * (0.866)
    sin_shift = sin * (-0.5) + cos * (0.866)
    c = (cos_shift * dq0[0] -
         sin_shift * dq0[1] +
         dq0[2])

    return [a, b, c]


def dq0ToabcCosSinPowerInvariant(dq0, cossin):
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
    temp = dq0_to_abc_cos_sin(dq0, cossin)
    return constMult(temp, 1.224744871391589)


def constMult(arr, mag):
    """
    Performs an element based multiplication of the arr by the constant mag
    ie [arr[0]*mag, arr[1]*mag .... arr[n]*mag]

    :param arr: the list
    :param mag: The constant to multiply by

    :return abc: The transformed space in the abc frame
    """
    return tuple([k * mag for k in arr])


def abcTodq0(abc, theta):
    """
    Transforms from abc frame to the dq0 frame using the provided angle

    :param abc: The values in the abc reference frame
    :param theta: The angle [radians]

    :return dq0: The transformed space in the abc frame
    """
    return abcTodq0CosSin(abc, thetatoCossine(theta))


def abcTodq0CosSin(abc, cossin):
    """
    Transforms from abc frame to the dq0 frame using the provided cos-sin
    This implementation tries to improve by utilising the provided cos-sin
    to minimise calls to calculate the cossine etc

    :param abc: The values in the abc reference frame
    :param cossine: The cossine of the angle [cos(theta), sine (theta)]

    :return dq0: The transformed space in the abc frame
    """
    # Seperate sine and cosine elements
    cos = cossin[0]
    sin = cossin[1]
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

    return [d, q, z]


def abctoAlphaBeta(abc):
    """
    Transforms from abc frame to the alpha-beta frame

    :param abc: The values in the abc reference frame

    :return [alpha,beta]: The transformed alpha beta results
    """

    alpha = (2 / 3) * (abc[0] - 0.5 * abc[1] - 0.5 * abc[2])
    beta = (2 / 3) * (0.866 * abc[1] - 0.866 * abc[2])

    return [alpha, beta]


def thetatoCossine(theta):
    """
    Transforms from provided angle to the relavent cossine values

    :param theta: The angle [In RADIANS]

    :return [alpha,beta]: The resultant cossine
    """
    alpha = math.cos(theta)
    beta = math.sin(theta)
    return [alpha, beta]


def inst_rms(arr):
    return (math.sqrt(arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2])) / 1.732050807568877


def inst_power(varr, iarr):
    return (varr[0] * iarr[0] + varr[1] * iarr[1] + varr[2] * iarr[2])


def inst_reactive(varr, iarr):
    vlines = phaseToLines(varr)
    return -(vlines[1] * iarr[0] + varr[2] * iarr[1] + varr[0] * iarr[2]) * 0.5773502691896258


def phaseToLines(varr):
    return (varr[0] - varr[1], varr[1] - varr[2], varr[2] - varr[0])
