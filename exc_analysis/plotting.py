#! /usr/bin/env python

##########################################################################################
# plotting.py
#
# Plotting functions for jupyter notebook research
#
# NOTE:
#
# Created: February 17, 2017
#   - Mitchell Allain
#   - allain.mitch@gmail.com
#
# Modified:
#   * April 04, 2017 - added to package exc_analysis
#
##########################################################################################

import numpy as np
import math
from kinematics import exc
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


labels = ['Boom', 'Stick', 'Bucket', 'Swing']


def draw_exc(ax, state, lw=2, lock_axes=True, rotate=True, gnd_offset=17.1):
    '''Draws the excavator on 3D xyz axes with lines for each linkage
        uses random colors for each drawing

    Args:
        ax (matplotlib.Axis):  the axis to draw on
        state (array-like): 1-D state vector with length 4 (bm, sk, bk, sw)
        lw (int): linewidth for matplotlib

    Returns:
         none
    '''
    t1 = state[3]
    # Define lengths
    a1 = exc['a1']
    a2 = exc['a2']
    a3 = exc['a3']
    a4 = exc['a4']
    # Compute or Get joint angles
    # Boom angle
    r_c1 = state[0] + exc['r_cyl1']
    a_a1b = np.arccos((exc['r_o1b']**2 + exc['r_o1a']**2 - r_c1**2)/(2 * exc['r_o1b']*exc['r_o1a']))
    t2 = a_a1b - exc['a_b12'] - exc['a_a1x1']

    # Stick angle
    r_c2 = state[1] + exc['r_cyl2']
    a_c2d = np.arccos((exc['r_o2c']**2 + exc['r_o2d']**2 - r_c2**2)/(2 * exc['r_o2c'] * exc['r_o2d']))
    t3 = 3 * np.pi - exc['a_12c'] - a_c2d - exc['a_d23']

    # Bucket angle
    r_c3 = state[2] + exc['r_cyl3']
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

    gos = gnd_offset

    gnd = np.array([0]*3)

    base = np.array([0, 0, gos])

    o1 = np.array([a1*c1, a1*s1, gos])

    o2 = np.array([(a2*c2 + a1)*c1,
                  (a2*c2 + a1)*s1,
                  a2*s2 + gos])

    o3 = np.array([(a2*c2 + a3*c23 + a1)*c1,
                  (a2*c2 + a3*c23 + a1)*s1,
                  gos + a2*s2 + a3*s23])

    o4 = np.array([(a4*c234 + a3*c23 + a2*c2 + a1)*c1,
                   (a4*c234 + a3*c23 + a2*c2 + a1)*s1,
                   gos + a2*s2 + a3*s23 + a4*s234])

    l0 = zip(gnd, base)
    l1 = zip(base, o1)
    l2 = zip(o1, o2)
    l3 = zip(o2, o3)
    l4 = zip(o3, o4)

    # color = np.random.rand(3)

    for line in [l0, l1, l2, l3, l4]:
        ax.plot(line[0], line[1], line[2], '-o', zdir='z', linewidth=lw, c='y', zorder=0)

    if lock_axes:
        ax.set_xlim3d([0, 80])
        ax.set_ylim3d([0, 80])
        ax.set_zlim3d([0, 50])

    if rotate:
        ax.view_init(azim=-137, elev=35)

    return


def orient_plot(ax):
    ax.set_xlim3d([-20, 80])
    ax.set_ylim3d([0, 80])
    ax.set_zlim3d([0, 50])
    ax.view_init(azim=-142, elev=14)


def plot_3d_scatter(states, title, color='r'):
    ''' 3D plot labeled clusters

    Args:
        states (np.array): n x 3, n samples by xyz coordinates of end-effector position
        title (str): string for plot title
        color (str or 2D array): color for all points

    Returns:
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(states[:, 0], states[:, 1], states[:, 2], zdir='z', c=color)

    plt.title(title)
    orient_plot(ax)


def plot_3d_labeled_clusters(states, labels, title, colors):
    ''' 3D plot labeled clusters

    Args:
        states (np.array): n x 3, n samples by xyz coordinates of end-effector position
        labels (np.array): n length vector of labels corresponding to states
        title (str): string for plot title
        colors (itertools.cycle): color cycle

    Returns:
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in np.unique(labels):
        # if not np.any(labels == i):
        #     continue
        ax.scatter(states[labels == i, 0], states[labels == i, 1], states[labels == i, 2], zdir='z', c=colors.next())

    plt.title(title)


def get_color_cycle(ind):
    ''' Return an itertools.cycle color cycle '''

    # Some colorsets
    csets = [['navy', 'c', 'cornflowerblue', 'gold', 'darkorange', 'r']]

    return itertools.cycle(csets[ind])


def view_trial(trial, trial_type='manual'):
    ''' View position data from a pandas dataframe

    Args:
        trial (pandas.dataframe): a dataframe with at minimum the measurement data '''

    if trial_type == 'manual':
        signals = ['Ms', 'Cmd']
    elif trial_type == 'blended':
        signals = ['Ms', 'Cmd', 'Ctrl', 'Blended']
    elif trial_type == 'autonomous':
        signals = ['Ms', 'Cmd', 'Error']

    dim = len(signals)

    fig, ax = plt.subplots(nrows=dim, sharex=True, figsize=(6, 2*dim))

    for i in range(dim):
        # Change the axis units to serif
        plt.setp(ax[i].get_ymajorticklabels(), family='serif', fontsize=14)
        plt.setp(ax[i].get_xmajorticklabels(), family='serif', fontsize=14)

        ax[i].spines['right'].set_color('none')
        ax[i].spines['top'].set_color('none')

        ax[i].xaxis.set_ticks_position('bottom')
        ax[i].yaxis.set_ticks_position('left')

        # Turn on the plot grid and set appropriate linestyle and color
        ax[i].grid(True, linestyle=':', color='0.75')
        ax[i].set_axisbelow(True)

    X = np.repeat(np.expand_dims(trial['Time'].values, 1), dim, 1)

    for i in xrange(dim):
        Y = trial[[lbl + ' ' + signals[i] for lbl in labels]].values
        lines = ax[i].plot(X, Y, linewidth=2, linestyle='-')

        plt.figlegend(lines, labels=labels, loc='lower right', fontsize=12)

    plt.gcf().subplots_adjust(bottom=0.5, left=0.5)

    plt.tight_layout()


def view_assistance_magnitude(blended):
    ''' View assistance as a stacked plot '''
    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(6, 8))

    for i in range(4):
        # Change the axis units to serif
        plt.setp(ax[i].get_ymajorticklabels(), family='serif', fontsize=14)
        plt.setp(ax[i].get_xmajorticklabels(), family='serif', fontsize=14)

        ax[i].spines['right'].set_color('none')
        ax[i].spines['top'].set_color('none')

        ax[i].xaxis.set_ticks_position('bottom')
        ax[i].yaxis.set_ticks_position('left')

        # Turn on the plot grid and set appropriate linestyle and color
        ax[i].grid(True, linestyle=':', color='0.75')
        ax[i].set_axisbelow(True)

    X = blended['Time'].values

    for i in range(4):
        ax[i].stackplot(X, blended[labels[i] + ' Cmd'], blended[labels[i] + ' Blended'])
