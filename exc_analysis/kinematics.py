#! /usr/bin/env python

##########################################################################################
# kinematics.py
#
# Compute forward kinematics and other kinematic functions
#
# NOTE: part of exc_analysis package
#
# Created: April 04, 2017
#   - Mitchell Allain
#   - allain.mitch@gmail.com
#
# Modified:
#   *
#
##########################################################################################

import numpy as np
import math
import numpy.linalg as linalg


## Coming directly from exc.mat via mat4py
exc = {u'a_g34': 1.6587914336784744,
       u'lc4': 76.42,
       u'lc2': 24.1,
       u'lc3': 9.62,
       u'ucwidth': 25,
       u'r_bc': 5.7,
       u'r_o2b': 29.2,
       u'r_gh': 6,
       u'baserpm': 27,
       u'motorrpm': 7000,
       u'r_o4g': 12.8,
       u'r_fh': 6,
       u'r_de': 9.4,
       u'r_df': 29,
       u'r_o1b': 22,
       u'ps': 290,
       u'a_dfe': 0.2119681622609662,
       u'r_o1a': 7.8,
       u'ucweight': 7.091,
       u'I3': 0.00398,
       u'I2': 0.01143,
       u'r_o3d': 33,
       u'I4': 0.00059,
       u'r_o3f': 4,
       u'r_o3g': 4.7,
       u'm4': 0.27,
       u'm3': 0.5829,
       u'm2': 0.7415,
       u'uclength': 38.3,
       u'a_a1x1': 0.8160994608031792,
       u'a_23d': 0.08703188367881715,
       u'r_o1c': 27.5,
       u'a1': 3.7,
       u'a_12c': 0.48263391433370934,
       u'a3': 25.3,
       u'a2': 48.2,
       u'a4': 11.5,
       u'r_ef': 21.2,
       u'r_cyl3': 15.55,
       u'r_o2f': 21.3,
       u'r_cyl1': 16.3,
       u'r_o2d': 8.1,
       u'r_o2c': 26.7,
       u'ucheigth': 9.5,
       u'a_b12': 0.398527674842184,
       u'a_d23': 2.779612365176967,
       u'r_cyl2': 19.4,
       u'r_o2e': 7}


def forward_kin_pt(exc, sw, bm, sk, bk, bias=0):
    '''This func is the same as 'forward_kin' but is easily vectorized.

    Note: ported to Python from MATLAB "fwd_kin.m", assumed options = [0, 0]

    Example:


    Args:
        exc (dict): a dict of the excavator physical parameters
        sw (float): the swing angle
        bm (floats): boom displacement in cm
        sk      ^^
        bk      ^^
        bias (float): positive z bias on output, to adjust weird base frame, (i.e. 17.1cm)

    Returns:
        eef (list: float): the position of the end-effector (EEF) in (x, y, z - base frame) and the angle of the bucket (axis x4 w.r.t. x1(0?) ground axis)
    '''
    # Assign the base swing angle
    t1 = sw

    # Define lengths
    a1 = exc['a1']
    a2 = exc['a2']
    a3 = exc['a3']
    a4 = exc['a4']

    # Compute or Get joint angles
    # Boom angle
    r_c1 = bm + exc['r_cyl1']
    a_a1b = np.arccos((exc['r_o1b']**2 + exc['r_o1a']**2 - r_c1**2)/(2 * exc['r_o1b']*exc['r_o1a']))
    t2 = a_a1b - exc['a_b12'] - exc['a_a1x1']

    # Stick angle
    r_c2 = sk + exc['r_cyl2']
    a_c2d = np.arccos((exc['r_o2c']**2 + exc['r_o2d']**2 - r_c2**2)/(2 * exc['r_o2c'] * exc['r_o2d']))
    t3 = 3 * np.pi - exc['a_12c'] - a_c2d - exc['a_d23']

    # Bucket angle
    r_c3 = bk + exc['r_cyl3']
    a_efh = np.arccos((exc['r_ef']**2 + exc['r_fh']**2 - r_c3**2)/(2 * exc['r_ef'] * exc['r_fh']))
    a_hf3 = np.pi - exc['a_dfe'] - a_efh
    r_o3h = math.sqrt(exc['r_o3f']**2 + exc['r_fh']**2 - 2 * exc['r_o3f'] * exc['r_fh'] * np.cos(a_hf3))
    a_f3h = np.arccos((r_o3h**2 + exc['r_o3f']**2 - exc['r_fh']**2)/(2 * r_o3h * exc['r_o3f']))
    a_h3g = np.arccos((r_o3h**2 + exc['r_o3g']**2 - exc['r_gh']**2)/(2 * r_o3h * exc['r_o3g']))
    t4 = 3 * np.pi - a_f3h - a_h3g - exc['a_g34'] - exc['a_23d']

    c1 = np.cos(t1)
    c2 = np.cos(t2)
    c234 = np.cos(t2 + t3 + t4)
    c23 = np.cos(t2 + t3)
    s1 = np.sin(t1)
    s2 = np.sin(t2)
    s234 = np.sin(t2 + t3 + t4)
    s23 = np.sin(t2 + t3)

    # Transformation matrices from every frame to base frame
    # T01 = np.array([[c1, 0, s1, a1 * c1],
    #                 [s1, 0, -c1, a1 * s1],
    #                 [0, 1, 0, 0],
    #                 [0, 0, 0, 1]])

    # T02 = np.array([[c1*c2, -c1*s2, s1, c1*(a2*c2+a1)],
    #                 [s1*c2, -s1*s2, -c1, s1*(a2*c2+a1)],
    #                 [s2, c2, 0, a2*s2],
    #                 [0, 0, 0, 1]])

    # T03 = np.array([[c1*c23, -c1*s23, s1, c1*(a3*c23+a2*c2+a1)],
    #                 [s1*c23, -s1*s23, -c1, s1*(a3*c23+a2*c2+a1)],
    #                 [s23, c23, 0, a3*s23+a2*s2],
    #                 [0, 0, 0, 1]])

    P04 = np.array([[c1*(a4*c234+a3*c23+a2*c2+a1)],
                    [s1*(a4*c234+a3*c23+a2*c2+a1)],
                    [(a4*s234+a3*s23+a2*s2)],
                    [1]])

    # print(P04)

    # Bucket angle; angle between x4 and x0-y0 plane
    tb = t2 + t3 + t4 - 3 * np.pi
    # T04 = np.array([[np.cos(tb) * c234, -np.cos(tb)*s234, np.sin(tb), P04[0]],
    #                 [np.sin(tb) * c234, -np.sin(tb)*s234, np.cos(tb), P04[1]],
    #                 [s234, c234, 0, P04[2]],
    #                 [0, 0, 0, P04[3]]])

    # Position and orientation of the end effector
    eef = [axis.pop() for axis in P04[0:3].tolist()]
    assert eef
    eef.append(tb)

    return eef[0], eef[1], eef[2] + bias


def forward_kin_array(states, bias=17.1):
    ''' Converts a numpy array of states in the actuator space to end-effector position
        in the xyz cartesian space, using above func forward_kin_pt()

    Args:
        states (np.array): n x m array of n samples by m=4 actuators
        bias (float): see bias in forward_kin_pt()

    Returns:
        states_xyz (np.array): n x 3 array of n samples in xyz workspace
    '''

    forward_kin_vec = np.vectorize(forward_kin_pt)
    states_xyz = np.array(forward_kin_vec(exc, states[:, 3], states[:, 0],
                                          states[:, 1], states[:, 2], bias=bias))

    return states_xyz.T
