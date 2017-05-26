#! /usr/bin/env python

##########################################################################################
# prediction.py
#
# Offline versions of prediction techniques to run over trial data
#
# NOTE:
#
# Created: May 23, 2017
#   - Mitchell Allain
#   - allain.mitch@gmail.com
#
# Modified:
#   *
#
# Todo:
#   *
#
##########################################################################################

import numpy as np
import pickle
from scipy.stats import mvn
from scipy.stats import multivariate_normal
import math
import logging
# import datetime
# import os
# import pdb

# LOG
logging.basicConfig(filename='prediction.log', level=logging.DEBUG)


class GaussianPredictor():
    '''Abstract class for predictors, subclass and implement an update method

    NOTE: Subgoal numbering now indexes from zero

    Args:
        filename (str): filename of pickled means, covs, queues, and transition matrix
        ll_func (func): function which accepts state and action vectors and
            means and cov arrays and returns a vector of likelihoods for each subgoal

    Attributes:
        last_confirmed (int): index of last subgoal from check_if_terminated()
        last_suspected (int): last subgoal to activate blending
        subgoal (int): current subgoal
        likelihood (np.array): k-length array of action likelihoods
        subgoal_probability (np.array): k-length array of posterior probabilities
        alpha (float): blending parameter
        alpha_threshold (float): threshold to activate blending
        means (np.array): k x m array of subgoal means
        covs (np.array): k x m x m array of subgoal covariances
        trans (np.array): k x k stochastic transition matrix
        kdim (np.array): number of subgoals inferred from the above vars

    Methods:
        update(): maps the values of subgoal_probability to a blending parameter value
        get_target_sg_pos(): returns location of current subgoal
        check_if_terminated(): checks if current state is within subgoal dist
                               with confidence over threshold
    '''
    def __init__(self, filename='gmm_model_exp.pkl'):
        self.last_confirmed = -1
        self.last_suspected = -1
        self.subgoal = -1
        self.alpha = 0
        self.alpha_threshold = 0.7
        # Model params
        with open(filename, 'rb') as openfile:
            tmp = pickle.load(openfile)
            self.means = tmp['means']
            self.covs = tmp['covs']
            self.trans = tmp['trans']
            # self.trans = np.ones((6, 6))
            self.queus = tmp['queues']
        self.kdim = len(self.means)
        self.subgoal_probability = np.zeros(self.kdim)
        self.likelihood = np.zeros(self.kdim)

    def update(self, states, actions):
        self.check_if_terminated(states)
        self.likelihood = get_mvn_action_likelihood_marginal(state, action, self.means, self.covs)[0]
        # self.likelihood = self.get_likelihood(states, actions, self.means, self.covs, **kwargs)

        # Apply transition vector corresponding to last confirmed sg
        self.subgoal_probability = self.trans[:, self.last_confirmed] * self.likelihood

        # Normalize posterior
        self.subgoal_probability = np.nan_to_num(self.subgoal_probability / np.sum(self.subgoal_probability))
        # self.subgoal_probability = np.nan_to_num(self.subgoal_probability)

        MAP = np.max(self.subgoal_probability)
        if (MAP > self.alpha_threshold):
            self.alpha = lin_map(MAP, self.alpha_threshold, 1, 0.3, 0.6)
            self.subgoal = np.argmax(self.subgoal_probability)
        else:
            self.alpha = 0

    def get_target_sg_pos(self):
        return self.means[self.subgoal]

    def check_if_terminated(self, states, threshold=0.6):
        ''' See if states is within termination region, assign last subgoal'''
        termination_probability = np.array([multivariate_normal(self.means[i], self.covs[i]).pdf(states) for i in xrange(self.kdim)])
        if (termination_probability > threshold).any():
            self.last_confirmed = np.argmax(termination_probability)

    def check_if_terminated_update_stats(self, states, actions, threshold=0.001):
        ''' Checks if we are in a subgoal, which is defined as being within a
            (NEW) subgoal distribution (VALID), and having zero velocity (STILL)

            If all three conditions True, then add to queue and recompute stats
            Can suppress self.update_stats() for static distributions

            Todo: revise these conditions (is zero velocity appropriate?)
            '''
        termination_probability = np.array([multivariate_normal(self.means[i], self.covs[i]).pdf(states) for i in xrange(self.kdim)])
        NEW = (np.argmax(termination_probability) != self.last_confirmed)
        VALID = (termination_probability > threshold).any()
        STILL = (actions == 0).all()
        if VALID and STILL and NEW:
            # Set last_confirmed to sg index
            self.last_confirmed = np.argmax(termination_probability)
            # Add location to corresponding queue
            self.queues[self.last_confirmed].append(states)
            self.update_stats()

    def update_stats(self):
        ''' Recalculate stats for last confirmed'''
        self.means[self.last_confirmed] = np.mean(self.queues[self.last_confirmed], axis=0)
        self.covs[self.last_confirmed] = np.cov(np.array(self.queues[self.last_confirmed]).T)


def get_mvn_action_likelihood_marginal_mvndst(states, actions, means, covs):
    ''' Rewriting the original multivariate action likelihood to marginalize out inactive vars
        uses Alan Genz's multivariate normal Fortran function 'mvndst' in Scipy

    Args:
        states (np.array): m-length state or n x m array of states
        actions (np.array): m-length action or n x m array of actions
        means (np.array): k x m array of means for k subgoals
        covs (np.array): k x m x m array of m covariance matrices for k subgoals

    Returns:
        action_likelihoods (np.array): n x k array of likelihoods for each subgoal for n states

    TODO:
        marginalize inactive variables by dropping covariances instead of computing whole domain
    '''

    if states.shape != actions.shape:
        raise ValueError('state and action args must have equal dimension.')

    elif states.ndim == 1:
        states = np.expand_dims(states, axis=0)
        actions = np.expand_dims(actions, axis=0)

    action_likelihoods = np.zeros((states.shape[0], means.shape[0]))
    indicator = np.zeros(action_likelihoods.shape)

    # For state, action pair index i
    for i in xrange(states.shape[0]):
        # Find active axes and skip if null input
        active = np.where(actions[i] != 0)[0]
        if active.size == 0:
            break

        # Else, compute mvn pdf integration for each subgoal
        # Bounds are shifted so that dist is zero mean
        for g in xrange(means.shape[0]):
            low = np.copy(states[i] - means[g])
            upp = np.copy(states[i] - means[g])
            infin = np.zeros(actions.shape[1])

            # Iterate through active indices and set low and upper bounds of ATD action-targeted domain
            # infin is an integer code used by func mvndst.f
            for j in xrange(actions.shape[1]):
                if actions[i, j] < 0:  # Negative action
                    infin[j] = 0
                elif actions[i, j] > 0:  # Postive action
                    infin[j] = 1
                else:
                    infin[j] = -1

            # Marginalize out inactive variables by dropping means and covariances
            corr = pack_covs(covs[g])
            # logging.info('Correlation coeff: %s \n'
            #              'Covariance matrix: %s \n'
            #              'Active: %s' % (corr, covs, active))

            _, action_likelihoods[i, g], indicator[i, g] = mvn.mvndst(low, upp, infin, corr)

            # if (indicator[i, g] == 1):
            #     logging.error('mvn.mvndst() failed with args: \n'
            #                   'low: %s \n upp: %s \n'
            #                   'infin: %s \n corr: %s \n' % (low, upp, infin, corr))
    return action_likelihoods


def pack_covs(covs):
    ''' To support Alan Genz's mvndst function; go read the documentation '''
    d = len(covs)
    corr = np.zeros((d*(d-1)/2))
    for i in range(d):
        for j in range(d):
            if (i > j):
                corr[j + ((i-1) * i) / 2] = covs[i, j]/(math.sqrt(covs[i, i] * covs[j, j]))
    return corr


# def cov_to_corr(cov):
#     corr = np.zeros(cov.shape)
#     for (i, j), val in [(i, j) for i in range(cov.shape[0]) for j in range(cov.shape[1])]:
#         corr[i, j] = cov[i, j] / math.sqrt(cov[i, i] * cov[j, j])
#     return corr


def get_mvn_action_likelihood_marginal(states, actions, means, covs):
    ''' Rewriting the original multivariate action likelihood to marginalize out inactive vars
        uses Alan Genz/Enthought Inc.'s multivariate normal Fortran functions in Scipy

    Args:
        states (np.array): m-length state or n x m array of states
        actions (np.array): m-length action or n x m array of actions
        means (np.array): k x m array of means for k subgoals
        covs (np.array): k x m x m array of m covariance matrices for k subgoals

    Returns:
        action_likelihoods (np.array): n x k array of likelihoods for each subgoal for n states

    TODO:
        marginalize inactive variables by dropping covariances instead of computing whole domain
    '''

    if states.shape != actions.shape:
        raise ValueError('state and action args must have equal dimension.')

    elif states.ndim == 1:
        states = np.expand_dims(states, axis=0)
        actions = np.expand_dims(actions, axis=0)

    action_likelihoods = np.zeros((states.shape[0], means.shape[0]))
    indicator = np.zeros(action_likelihoods.shape)

    # For state, action pair index i
    for i in xrange(states.shape[0]):
        # Find active axes and skip if null input
        active = np.where(actions[i] != 0)[0]
        if active.size == 0:
            break

        # Else, compute mvn pdf integration for each subgoal
        for g in xrange(means.shape[0]):
            low = np.zeros(active.shape)
            upp = np.copy(low)

            # Iterate through active indices and set low and upper bounds of ATD action-targeted domain
            # Bounds at +/-10 sig figs because no option for infinite, check this is sufficient b/c skew
            for num, j in enumerate(active):
                if actions[i, j] < 0:  # Negative action
                    low[num] = means[g, j] - 10 * covs[g, j, j]
                    upp[num] = states[i, j]
                else:  # Postive action
                    low[num] = states[i, j]
                    upp[num] = means[g, j] + 10 * covs[g, j, j]

            # Marginalize out inactive variables by dropping means and covariances
            means_marg = means[g][active]
            covs_marg = covs[g][active][:, active]
            # pdb.set_trace()
            action_likelihoods[i, g], indicator[i, g] = mvn.mvnun(low, upp, means_marg, covs_marg, maxpts=100000)

            if (indicator[i, g] == 1):
                logging.info('{} {} {} {}'.format(low, upp, means_marg, covs_marg))
    # if (indicator == 1).any():
        # print('mvnun failed: error code 1')
        # print(low, upp, means, covs)
        # raise ArithmeticError('Fortran function mvnun: error code 1')
    return action_likelihoods[0]


def get_mvn_action_likelihood(states, actions, means, covs):
    ''' Taken from jupyter notebook gaussian-likelihood,
        uses Alan Genz/Enthought Inc.'s multivariate normal Fortran functions in Scipy

    Args:
        states (np.array): m-length state or n x m array of states
        actions (np.array): m-length action or n x m array of actions
        means (np.array): k x m array of means for k subgoals
        covs (np.array): k x m x m array of m covariance matrices for k subgoals

    Returns:
        action_likelihoods (np.array): n x k array of likelihoods for each subgoal for n states

    TODO:
        marginalize inactive variables by dropping covariances instead of computing whole domain
    '''
    if states.shape != actions.shape:
        raise ValueError('state and action args must have equal dimension.')
    elif states.ndim == 1:
        states = np.expand_dims(states, axis=0)
        actions = np.expand_dims(actions, axis=0)
    action_likelihoods = np.zeros((states.shape[0], means.shape[0]))
    indicator = np.zeros(action_likelihoods.shape)
    for i in xrange(states.shape[0]):
        for g in xrange(means.shape[0]):
            low = np.zeros(states.shape[1])
            upp = np.copy(low)
            for j in xrange(states.shape[1]):
                if actions[i, j] < 0:
                    low[j] = means[g, j] - 10 * covs[g, j, j]
                    upp[j] = states[i, j]
                elif actions[i, j] > 0:
                    low[j] = states[i, j]
                    upp[j] = means[g, j] + 10 * covs[g, j, j]
                else:  # Yields probability 1
                    low[j] = means[g, j] - 10 * covs[g, j, j]
                    upp[j] = means[g, j] + 10 * covs[g, j, j]
            # pdb.set_trace()
            action_likelihoods[i, g], indicator[i, g] = mvn.mvnun(low, upp, means[g], covs[g], maxpts=100000)
            if (indicator[i, g] == 1):
                logging.info('{} {} {} {}'.format(low, upp, means, covs))
    # if (indicator == 1).any():
        # print('mvnun failed: error code 1')
        # print(low, upp, means, covs)
        # raise ArithmeticError('Fortran function mvnun: error code 1')
    return action_likelihoods


def get_action_comp_likelihood(states, actions, means, covs, beta=1):
    ''' Returns the exponential likelihoods for each subgoal, when the direction
        of the action is compared to the vector to the subgoal

            P(ai | si, zi) = exp[ beta * (cos(theta) - 1) ]

        where theta is angle between action and vec to subgoal
    '''
    p_sz = means - states
    dir_comp = (np.dot(p_sz, actions) / (np.linalg.norm(p_sz, axis=1) * np.linalg.norm(actions))) - 1
    return np.e**(beta * dir_comp)


class TriggerPrediction():
    '''The trigger prediction class uses task specific event triggers to determine the current subgoal

    Args:
        sg_model (list: dicts): list of subgoal model dicts, see example below
        mode (int): 0 only terminates, 1 is IFAC style
        alpha (float): BSC blending parameter preset when active, can turn off with 0

    Example arg:
        sg_model = [{'subgoal': 1,
                     'it': [3, -0.5]                            * Joystick index 3 (swing) move past halfway left
                     'subgoal_pos': [6.75, 0.91, 9.95, 1.41]    * Over the pile (actuator space coordinates)
                     'npt': [3, 3, 3, 0.2]}                     * +/- each of these values forms boundary around subgoal
                     'onpt': []},                               * Not yet implemented

                    {'subgoal': 2, ...
                    ...}]
    Attributes:
        alpha (float): BSC blending parameter alpha
        subgoal (int): current triggered subgoal
        prev (int): previous subgoal index
        regen (bool): flag to regenerate trajectories
        active (bool): assistance active
    '''
    def __init__(self, sg_model, mode=1, alpha=0):
        self.mode = mode
        self.alpha = alpha
        self.sg_model = sg_model

        self.dispatch = {0: self.update_0,
                         1: self.update_1}

        self.subgoal = 0            # Subgoal 0 denotes no subgoal to start
        self.prev = 6
        self.active = False         # Active is bool, False means no assistance to start
        self.regen = True

        # Important: build a list of the subgoals for iterating
        self.sg_list = [self.sg_model[i]['subgoal'] for i in range(len(self.sg_model))]

    def update(self, state, action):
        ''' General update dispatch function
        Args:
            state (np.array): m-length array of measurements
            action (np.array): m-length array of normalized (and deadbanded) inputs
        '''
        return self.dispatch[self.mode](state, action)

    def update_0(self, state, action):
        ''' Mode 0: only terminates, no active'''
        # Look for a terminating cue
        for sg in self.sg_model:
            # Are we in a termination set?
            termination = ([abs(state[i] - sg['subgoal_pos'][i]) < sg['npt'][i] for i in range(4)] == [True]*4)

            # Is this termination set different from our previous subgoal termination?
            # I.e., we don't want to reterminate in the same set over and over.
            different = (sg['subgoal'] != self.prev)

            if termination and different:
                print('Terminated: ', sg['subgoal'])
                self.prev = sg['subgoal']
                self.subgoal = (sg['subgoal'] % len(self.sg_list)) + 1  # i = (i % length) + 1 (some magic)
                self.regen = True
                self.active = False

    def update_1(self, state, action):
        ''' Mode 1: IFAC style FSM predictor '''
        # pdb.set_trace()
        # Look for a terminating cue
        for sg in self.sg_model:
            # Are we in a termination set?
            termination = ([abs(state[i] - sg['subgoal_pos'][i]) < sg['npt'][i] for i in range(4)] == [True]*4)

            # Is this termination set different from our previous subgoal termination?
            # I.e., we don't want to reterminate in the same set over and over.
            different = (sg['subgoal'] != self.prev)

            if termination and different:
                # print('Terminated: ', sg['subgoal'])
                self.prev = sg['subgoal']
                self.subgoal = (sg['subgoal'] + 1) % len(self.sg_list)  # i = (i % length) + 1 (some magic)
                self.regen = True
                self.active = False

        less_than = ((action[self.sg_model[self.subgoal-1]['it'][0]] < self.sg_model[self.subgoal-1]['it'][1]))
        # print less_than
        negative = ((self.sg_model[self.subgoal-1]['it'][1]) < 0)
        # print negative
        if not (less_than != negative):  # If input < threshold and threshold negative, or > = threshold and threshold positive
            self.active = True
            # print(self.subgoal, 'Ass: True')
        else:  # No input
            self.active = False

        return self.subgoal, self.active


def lin_map(x, a, b, u, v):
    ''' Maps x from interval (A, B) to interval (a, b)
    TODO: make multidimensional'''
    return (x - a) * ((v - u) / (b - a)) + u


# def construct_prediction_model()
class ActionCompPredictor():
    '''Action comparison based predictor

    NOTE: Subgoal numbering now indexes from zero

    Args:
        filename (str): filename of pickled means, covs, queues, and transition matrix

    Attributes:
        last_confirmed (int): index of last subgoal from check_if_terminated()
        last_suspected (int): last subgoal to activate blending
        subgoal (int): current subgoal
        likelihood (np.array): k-length array of action likelihoods
        subgoal_probability (np.array): k-length array of posterior probabilities
        alpha (float): blending parameter
        alpha_threshold (float): threshold to activate blending
        means (np.array): k x m array of subgoal means
        covs (np.array): k x m x m array of subgoal covariances
        trans (np.array): k x k stochastic transition matrix
        kdim (np.array): number of subgoals inferred from the above vars

    Methods:
        update(): maps the values of subgoal_probability to a blending parameter value
        get_target_sg_pos(): returns location of current subgoal
        check_if_terminated(): checks if current state is within subgoal dist
                               with confidence over threshold
    '''
    def __init__(self, filename='gmm_model_exp.pkl'):
        self.last_confirmed = -1
        self.last_suspected = -1
        self.subgoal = -1
        self.alpha = 0
        self.alpha_threshold = 0.7

        # Model params
        with open(filename, 'rb') as openfile:
            tmp = pickle.load(openfile)
            self.means = tmp['means']
            self.covs = tmp['covs']
            self.trans = tmp['trans']
            self.queus = tmp['queues']

        self.kdim = len(self.means)
        self.subgoal_probability = np.zeros(self.kdim)
        self.likelihood = np.zeros(self.kdim)

    def update(self, states, actions):
        self.check_if_terminated(states)

        # Action comparison likelihood
        self.likelihood = get_action_comp_likelihood(states, actions, self.means,
                                                     self.covs, **kwargs)

        # Apply transition vector corresponding to last confirmed sg
        self.subgoal_probability = self.trans[:, self.last_confirmed] * self.likelihood

        # Normalize posterior
        self.subgoal_probability = np.nan_to_num(self.subgoal_probability / np.sum(self.subgoal_probability))
        # self.subgoal_probability = np.nan_to_num(self.subgoal_probability)

        p_map = np.max(self.subgoal_probability)
        if (p_map > self.alpha_threshold):
            self.alpha = lin_map(p_map, self.alpha_threshold, 1, 0.3, 0.6)
            self.subgoal = np.argmax(self.subgoal_probability)
        else:
            self.alpha = 0

    def get_target_sg_pos(self):
        return self.means[self.subgoal]

    def check_if_terminated(self, states, threshold=0.6):
        ''' See if states is within termination region, assign last subgoal'''
        termination_probability = np.array([multivariate_normal(self.means[i], self.covs[i]).pdf(states) for i in xrange(self.kdim)])
        if (termination_probability > threshold).any():
            self.last_confirmed = np.argmax(termination_probability)

    def check_if_terminated_update_stats(self, states, actions, threshold=0.001):
        ''' Checks if we are in a subgoal, which is defined as being within a
            (NEW) subgoal distribution (VALID), and having zero velocity (STILL)

            If all three conditions True, then add to queue and recompute stats
            Can suppress self.update_stats() for static distributions

            Todo: revise these conditions (is zero velocity appropriate?)
            '''
        termination_probability = np.array([multivariate_normal(self.means[i], self.covs[i]).pdf(states) for i in xrange(self.kdim)])
        NEW = (np.argmax(termination_probability) != self.last_confirmed)
        VALID = (termination_probability > threshold).any()
        STILL = (actions == 0).all()
        if VALID and STILL and NEW:
            # Set last_confirmed to sg index
            self.last_confirmed = np.argmax(termination_probability)
            # Add location to corresponding queue
            self.queues[self.last_confirmed].append(states)
            self.update_stats()

    def update_stats(self):
        ''' Recalculate stats for last confirmed'''
        self.means[self.last_confirmed] = np.mean(self.queues[self.last_confirmed], axis=0)
        self.covs[self.last_confirmed] = np.cov(np.array(self.queues[self.last_confirmed]).T)


def get_action_comp_likelihood(states, actions, means, covs, beta=1):
    ''' Returns the exponential likelihoods for each subgoal, when the direction
        of the action is compared to the vector to the subgoal

            P(ai | si, zi) = exp[ beta * (cos(theta) - 1) ]

        where theta is angle between action and vec to subgoal
    '''
    p_sz = means - states
    dir_comp = (np.dot(p_sz, actions) / (np.linalg.norm(p_sz, axis=1) * np.linalg.norm(actions))) - 1
    return np.e**(beta * dir_comp)
