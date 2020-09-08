#!/usr/bin/env python
# referenced Pedro Santana's original belief.py for his package of rao*
# slimmed it down and simplify

# author: Yun Chang
# yunchang@mit.edu

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull


sigma_w = 0.07
sigma_v = 0.05
sigma_b0 = 0.01
agent_sigma_b0 = 0.05

sigma_w = 0.03
sigma_v = 0.02

deltaT = 0.5
m = 10
b = 50

# Dynamics
n = 4
Ax = np.matrix([[1, deltaT], [0, 1 - deltaT * b / m]])
Bx = np.matrix([[0], [deltaT / m]])
Ay = np.matrix([[1, deltaT], [0, 1 - deltaT * b / m]])
By = np.matrix([[0], [deltaT / m]])

A = np.matrix([[Ax[0, 0], 0, Ax[0, 1], 0],
               [0,  Ay[0, 0], 0, Ay[0, 1]],
               [0, Ax[1, 0], Ax[1, 1], 0],
               [0, 0, Ay[1, 0], Ay[1, 1]]])

B = np.matrix([[0, 0], [0, 0], [Bx[1, 0], 0], [0, By[1, 0]]])

C = np.eye(n)
Bw = np.eye(n)
Dv = np.eye(n)

K = np.matrix([[0, 0, 29.8815, 0],
               [0, 0, 0, 29.8815]])

Ac = A + B * K
K0 = np.diag([(1 - Ac[2, 2]) / B[2, 0], (1 - Ac[3, 3]) / B[3, 1]])
# print('k0', K0)

r = [[1], [1]]

# print('\nA:\n', A)
# print('\nB:\n', B)

a1 = 2
bb1 = -2
a2 = 1
bb2 = 2
a3 = -4
bb3 = 27
a4 = -1 / 3
bb4 = 5


class ContinuousBeliefState(object):
    """
    Class representing a continuous belief state.
    """

    def __init__(self, x=0, y=0, v_x=0, v_y=0, decimals=5):
        self.n = 4
        self.nn = self.n * 2
        self.mean_b = np.zeros([self.nn, 1])
        self.mean_b[0] = x
        self.mean_b[1] = y
        self.mean_b[2] = v_x
        self.mean_b[3] = v_y
        self.belief = self

        self.agent_mean_b = np.zeros([self.nn, 1])
        self.agent_sigma_b = np.zeros([self.nn, self.nn])

        self.sigma_b = np.zeros([self.nn, self.nn])
        for i in range(self.n):
            self.sigma_b[i][i] = sigma_b0
            self.agent_sigma_b[i][i] = agent_sigma_b0
        # print("\nmean_b0:\n", self.mean_b)
        # print("\nsigma_b0:\n", self.sigma_b)
        self.current_P = sigma_b0 * np.eye(n)
        self.previous_action = None

    def __len__(self):
        return 7

    def set_agent_coords(self, x, y):
        self.agent_mean_b[0] = x
        self.agent_mean_b[1] = y

    def state_print(self):
        return "ContBeliefState x:" + "{0:.2f}".format(float(self.mean_b[0])) + " y:" + "{0:.2f}".format(float(self.mean_b[1]))

    def copy(self):
        '''Return a copy of the this belief_state as a new object'''
        new_belief_state = ContinuousBeliefState()
        new_belief_state.mean_b = self.mean_b
        new_belief_state.sigma_b = self.sigma_b
        new_belief_state.current_P = self.current_P
        new_belief_state.agent_mean_b = self.agent_mean_b.copy()
        new_belief_state.agent_sigma_b = self.agent_sigma_b.copy()
        return new_belief_state


def cont_belief_update(belief_state, control_input):
    time_steps = 2

    # P = sigma_b0 * np.eye(n)
    # print(P)

    noise_covariance = np.diag(
        [sigma_w, sigma_w, sigma_w, sigma_w, sigma_v, sigma_v, sigma_v, sigma_v])

    current_belief = belief_state.copy()
    new_m_b = None
    new_sigma_b = None
    new_P = None

    for k in range(time_steps):

        m_b = current_belief.mean_b
        sigma_b = current_belief.sigma_b
        P = current_belief.current_P

        W = np.random.normal(0, sigma_w)
        V = np.random.normal(0, sigma_v)

        term_in_L = (A * P * A.T + Bw * sigma_w * np.eye(n) * Bw.T)
        L = term_in_L * C.T * (C * term_in_L * C.T +
                               Dv * sigma_v * np.eye(n) * Dv.T).I
        new_P = (np.eye(n) - L * C) * term_in_L
        # print('L', L)
        # print('P', P)

        # print(np.matrix([[A, B * K], 0]))
        # print(np.matrix([A, B * K]))
        # print([A, B * K])

        topAA = np.concatenate((A, B * K), 1)
        bottomAA = np.concatenate((L * C * A, A + B * K - L * C * A), 1)

        AA = np.concatenate((topAA, bottomAA), 0)
        # print(AA)

        BBu = np.concatenate((B, B), 0)
        # print(BBu)
        BBtop = np.concatenate((Bw, np.zeros([n, n])), 1)
        BBbottom = np.concatenate((L * C * Bw, L * Dv), 1)
        BB = np.concatenate((BBtop, BBbottom), 0)
        # print('BB', BB)

        new_m_b = AA * m_b + BBu * K0 * control_input
        new_sigma_b = AA * sigma_b * AA.T + BB * noise_covariance * BB.T

        current_belief.mean_b = new_m_b
        current_belief.sigma_b = new_sigma_b
        current_belief.current_P = new_P

        # print('m_b at ', k, ' is:\n', new_m_b)
        # print('sigma_b at ', k, ' is:\n', new_sigma_b)

    new_belief_state = ContinuousBeliefState()
    new_belief_state.mean_b = new_m_b
    new_belief_state.sigma_b = new_sigma_b
    new_belief_state.current_P = new_P
    new_belief_state.agent_mean_b = current_belief.agent_mean_b
    new_belief_state.agent_sigma_b = current_belief.agent_sigma_b

    return new_belief_state



def belief_update(m_b, sigma_b, K, K0, I, A, B, C, P, Bw, Dv, sigma_w, sigma_v, sigma_b0):

    noise_covariance = np.diag(
        [sigma_w, sigma_w, sigma_w, sigma_w, sigma_v, sigma_v, sigma_v, sigma_v])

    W = np.random.normal(0, sigma_w)
    V = np.random.normal(0, sigma_v)

    term_in_L = (A.dot(P).dot(A.T) + Bw.dot(sigma_w).dot(np.eye(n)).dot(Bw.T)).dot(C.T)

    term_in_L_2 = (C.dot(term_in_L).dot(C.T) +Dv.dot(sigma_v).dot(np.eye(n)).dot(Dv.T)).I

    L = term_in_L.dot(term_in_L_2)
    
    # print("L")
    # print(L)
    new_P = (np.eye(n) - L.dot(C)).dot(term_in_L)

    # print("new p")
    # print(new_P)


    topAA = np.concatenate((A, B.dot(K)), 1)
    bottomAA = np.concatenate((L.dot(C).dot(A), A + B.dot(K) - L.dot(C).dot(A)), 1)

    AA = np.concatenate((topAA, bottomAA), 0)
    # print("AA")
    # print(AA)

    BBu = np.concatenate((B, B), 0)
    BBtop = np.concatenate((Bw, np.zeros([n, n])), 1)
    BBbottom = np.concatenate((L.dot(C).dot(Bw), L.dot(Dv)), 1)
    BB = np.concatenate((BBtop, BBbottom), 0)
    # print("BB")
    # print(BB)

    new_m_b = AA.dot(m_b) + BBu.dot(K0).dot(I).T
    new_sigma_b = AA.dot(sigma_b).dot(AA.T) + BB.dot(noise_covariance).dot(BB.T)

    return new_m_b, new_sigma_b




def plot_belief_state(axis, belief_state, color=(1, 0, 0, 0.5)):
    new_ellipse = Ellipse(xy=(belief_state.mean_b[0], belief_state.mean_b[1]),
                          width=2 * np.sqrt(belief_state.sigma_b[0, 0]), height=2 * np.sqrt(belief_state.sigma_b[1, 1]), angle=0)
    new_ellipse.set_facecolor(color)
    axis.add_artist(new_ellipse)


def dynamic_obs_risk(ego_belief_state, obs_belief_state):
    ego_m = ego_belief_state.mean_b[0:2]
    ego_s = ego_belief_state.sigma_b[0:2, 0:2]
    obs_m = obs_belief_state.mean_b[0:2]
    obs_s = obs_belief_state.sigma_b[0:2, 0:2]
    x_mean, y_mean = ego_m - obs_m
    [[x_std, _], [_, y_std]] = np.sqrt(ego_s + obs_s)
    collision_box = 1
    x_risk = norm.cdf(collision_box, x_mean, x_std) - \
        norm.cdf(-collision_box, x_mean, x_std)
    y_risk = norm.cdf(collision_box, y_mean, y_std) - \
        norm.cdf(-collision_box, y_mean, y_std)
    return min(x_risk, y_risk)


def compute_risk(mu, std, obs_vertices):
    Hull = ConvexHull(obs_vertices)
    Eq = Hull.equations
    Ao = Eq[:,0:2]
    bo = -Eq[:,2]
    R = []
    for i in range(bo.size):
        R.append(compute_risk_edge(Ao[i,:], bo[i], mu, std))
    return float(min(R))

def compute_risk_edge(Ao, bo, mu, std):
    mu2 = Ao.dot(mu) - bo
    std2 = np.sqrt(Ao.dot(std*std).dot(Ao))
    return norm.cdf(0, mu2, std2) - norm.cdf(-np.inf, mu2, std2)
