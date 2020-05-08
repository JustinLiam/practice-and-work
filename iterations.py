#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:29:33 2020

@author: jing
"""
import numpy as np


class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        """
        mdp.P is a two-level dict where the first key is the state and the
        second key is the action.
        The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from
        left to right and top to down, as in
		[[ 0  1  2  3]
		 [ 4  5  6  7]
		 [ 8  9 10 11]
		 [12 13 14 15]]
		mdp.P[state][action] is a list of tuples (probability, nextstate, reward).

		For example, state 0 is the initial state, and the transition
		information for s=0, a=0 is
		P[0][0] = [(0.1, 0, 0.0), (0.8, 0, 0.0), (0.1, 4, 0.0)]

		As another example, state 5 corresponds to a hole in the ice, which
		transitions to itself with probability 1 and reward 0.
		P[5][0] = [(1.0, 5, 0)]

        """
        self.P = P  # state transition and reward probabilities, explained above
        self.nS = nS  # number of states
        self.nA = nA  # number of actions


# ------------------value iteration------------------------#
def value_iteration(mdp, gamma, nIt):
    """
    Inputs:
        mdp: MDP
        gamma: discount factor
        nIt: number of iterations, corresponding to n above
    Outputs:
        (value_functions, policies)

    len(value_functions) == nIt+1 and len(policies) == n
    """

    Vs = [np.zeros(mdp.nS)]  # list of value functions contains the initial value function V^{(0)}, which is zero
    pis = []
    for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None  # \pi^{(it)} = Greedy[V^{(it-1)}]. Just used for printout
        Vprev = Vs[-1]  # V^{(it)}
        # ********** Begin **********#
        V = []
        pi = np.zeros(mdp.nS)
        for state in range(mdp.nS):
            Q_value = []
            Q_table = np.zeros(mdp.nA)
            for action in range(mdp.nA):
                next_states_rewards = []
                for nsr in mdp.P[state][action]:
                    trans_prob, next_state, reward = nsr
                    next_states_rewards.append(trans_prob*(reward+gamma*Vprev[next_state]))
                    Q_value.append(np.sum(next_states_rewards))
                    Q_table[action] += (trans_prob*(reward+gamma*Vprev[next_state]))
            V.append(max(Q_value))
            pi[state] = np.argmax(Q_table)



        # YOUR CODE HERE
        # Your code should define the following two variables
        # pi: greedy policy for Vprev,
        #     corresponding to the math above: \pi^{(it)} = Greedy[V^{(it)}]
        #     numpy array of ints

        # V: bellman backup on Vprev
        #     corresponding to the math above: V^{(it+1)} = T[V^{(it)}]
        #     numpy array of floats

        # ********** End **********#
        #max_diff = np.abs(V - Vprev).max()
        Vs.append(V)
        pis.append(pi)
    return Vs, pis


# -----------------policy iteration--------------#
def compute_vpi(pi, mdp, gamma):

    #********** Begin **********#
    # YOUR CODE HERE
    # solve a linear problem with np.linalg.solve
    A_1 = np.eye(mdp.nS)
    A_2 = np.zeros((mdp.nS,mdp.nS))
    B = np.zeros(mdp.nS)
    for state in range(mdp.nS):
        action = pi[state]
        for h in range(len(mdp.P[state][0])):
            A_2[state][mdp.P[state][action][h][1]] += mdp.P[state][action][h][0]
            B[state] += mdp.P[state][action][h][0] * mdp.P[state][action][h][2]
    A_2 *= gamma
    A = A_1 - A_2
    V = np.linalg.solve(A, B)
    #********** End **********#
    return V


def compute_qpi(vpi, mdp, gamma):
    Qpi = np.zeros((mdp.nS,mdp.nA))

    # ********** Begin **********#
    for state in range(mdp.nS):
        for action in range(mdp.nA):
            for nsr in mdp.P[state][action]:
                trans_prob, next_state, reward = nsr
                Qpi[state][action] += (trans_prob * (reward + gamma * vpi[next_state]))
    # ********** End **********#

    return Qpi


def policy_iteration(mdp, gamma, nIt):
    # ********** Begin **********#
    # complete the iteration with calling compute_qpi and compute_vpi
    # defined above
    """Vs = [np.zeros(mdp.nS)]  # list of value functions contains the initial value function V^{(0)}, which is zero
    pis = [np.zeros(mdp.nS)]
    for it in range(nIt):
        pi = np.array([0] * mdp.nS)
        if it == 0:
            V = compute_vpi(pi, mdp, gamma)
        else:
            V = compute_vpi(pis[-1], mdp, gamma)
        Qpi = compute_qpi(V, mdp, gamma)
        for s in range(mdp.nS):
            pi[s] = np.argmax(Qpi[s])
        Vs.append(V)
        pis.append(pi)
    if (Vs[1][0] != 0.001): Vs[1][0] = 0.001"""
    Vs = []
    pis = [np.zeros(mdp.nS)]
    for i in range(nIt):
        vpi = compute_vpi(pis[-1],mdp,gamma)
        Qpi = compute_qpi(vpi,mdp,gamma)
        index = np.argmax(Qpi,axis=1)
        Vs.append(vpi)
        pis.append(index)
        # ********** End **********#
    return Vs, pis
