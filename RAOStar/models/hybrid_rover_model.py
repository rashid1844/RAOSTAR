#!/usr/bin/env python

# author: Sungkweon Hong
# email: sk5050@mit.edu
# Grid model as simple model to test rao star

import math
import numpy as np
import time
from belief import BeliefState
from scipy.stats import norm
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull

# from continuous_belief import *
from shapely.geometry import *

class HybridRoverModel(object):

    def __init__(self, goal_region=Polygon([[8,8],[10,8],[10,10],[8,10]]), obstacle = np.array([[4,4],[3,5],[1,3],[2,3]]), goal_num_sample=3, prob_sample_success=0.9, DetermObs=True):


        # Time step
        self.deltaT = 0.5

        # Dynamics
        self.n = 4
        self.m = 10
        self.b = 50
        self.Ax = np.matrix([[1, self.deltaT], [0, 1 - self.deltaT * self.b / self.m]])
        self.Bx = np.matrix([[0], [self.deltaT / self.m]])
        self.Ay = np.matrix([[1, self.deltaT], [0, 1 - self.deltaT * self.b / self.m]])
        self.By = np.matrix([[0], [self.deltaT / self.m]])

        self.A = np.matrix([[self.Ax[0, 0], 0, self.Ax[0, 1], 0],
                       [0,  self.Ay[0, 0], 0, self.Ay[0, 1]],
                       [0, self.Ax[1, 0], self.Ax[1, 1], 0],
                       [0, 0, self.Ay[1, 0], self.Ay[1, 1]]])

        self.B = np.matrix([[0, 0], [0, 0], [self.Bx[1, 0], 0], [0, self.By[1, 0]]])

        self.C = np.eye(self.n)
        self.Bw = np.eye(self.n)
        self.Dv = np.eye(self.n)

        # Covariance of uncertainties
        self.sigma_w = 0.01
        self.sigma_v = 0.01
        self.noise_covariance = np.diag([self.sigma_w, self.sigma_w, self.sigma_w, self.sigma_w, self.sigma_v, self.sigma_v, self.sigma_v, self.sigma_v])
        self.sigma_b0 = 0.01

        # Controller gain
        self.K = np.matrix([[0, 0, 29.8815, 0],
                       [0, 0, 0, 29.8815]])

        self.Ac = self.A + self.B * self.K
        self.K0 = np.diag([(1 - self.Ac[2, 2]) / self.B[2, 0], (1 - self.Ac[3, 3]) / self.B[3, 1]])

        # Control inputs
        self.v = 2  # velocity


        self.P = self.sigma_b0 * np.eye(self.n)

        # self.init_mu = np.zeros((2*n,1))
        # self.init_sigma = np.zeros((2*n,2*n))
        # self.init_sigma[0:4,0:4] = sigma_b0*np.eye(n)

        self.prob_sample_success = prob_sample_success
        self.action_list = ["Up","Left","Right","Down","Go_45","Sample"]
        self.obstacle = obstacle
        self.goal_region = goal_region
        self.goal_num_sample = goal_num_sample

        self.optimization = 'minimize'
        self.DetermObs = DetermObs

    def actions(self, state):
        return self.action_list

    def is_terminal(self, state):
        mu = state[0].get_mu()
        return Point(mu[0:2]).contains(self.goal_region) and state[1]>=self.goal_num_sample

    def state_transitions(self, state, action):

        if action=="Sample":
            m_b = state[0].get_mu()
            sigma_b = state[0].get_sigma()
            new_state = [[(GaussianState(mu=m_b, sigma=sigma_b), state[1]+1), self.prob_sample_success], [(GaussianState(mu=m_b, sigma=sigma_b), state[1]), 1-self.prob_sample_success]]
            return new_state
        
        elif action=="Down":
            I = self.v * np.array([0,-1])
        elif action=="Left":
            I = self.v * np.array([-1,0])
        elif action=="Right":
            I = self.v * np.array([1,0])
        elif action=="Go_45":
            I = self.v * np.array([1,1])
        elif action=="Up":
            I = self.v * np.array([0,1])

    
        m_b = state[0].get_mu()
        sigma_b = state[0].get_sigma()

        W = np.random.normal(0, self.sigma_w)
        V = np.random.normal(0, self.sigma_v)

        term_in_L = (self.A.dot(self.P).dot(self.A.T) + self.Bw.dot(self.sigma_w).dot(np.eye(self.n)).dot(self.Bw.T)).dot(self.C.T)
        term_in_L_2 = (self.C.dot(term_in_L).dot(self.C.T) +self.Dv.dot(self.sigma_v).dot(np.eye(self.n)).dot(self.Dv.T)).I
        L = term_in_L.dot(term_in_L_2)

        new_P = (np.eye(self.n) - L.dot(self.C)).dot(term_in_L)


        topAA = np.concatenate((self.A, self.B.dot(self.K)), 1)
        bottomAA = np.concatenate((L.dot(self.C).dot(self.A), self.A + self.B.dot(self.K) - L.dot(self.C).dot(self.A)), 1)

        AA = np.concatenate((topAA, bottomAA), 0)

        BBu = np.concatenate((self.B, self.B), 0)
        BBtop = np.concatenate((self.Bw, np.zeros([self.n, self.n])), 1)
        BBbottom = np.concatenate((L.dot(self.C).dot(self.Bw), L.dot(self.Dv)), 1)
        BB = np.concatenate((BBtop, BBbottom), 0)

        new_m_b = AA.dot(m_b) + BBu.dot(self.K0).dot(I).T
        new_sigma_b = AA.dot(sigma_b).dot(AA.T) + BB.dot(self.noise_covariance).dot(BB.T)

        new_state = [[(GaussianState(mu=new_m_b,sigma=new_sigma_b), state[1]), 1.0]]

        return new_state

    def observations(self, state):
        if self.DetermObs:
            return [(state, 1.0)]
        else:
            raise ValueError('observations should be deterministic!')


    def state_risk(self, state):
        mu = state[0].get_mu()
        sigma = state[0].get_sigma()
        risk = self.compute_risk(mu[0:2], sigma[0:2,0:2], self.obstacle)
        return risk

    def values(self,state,action):
        if action=="Sample":
            return 1.0
        else:
            return 1.0

    def heuristic(self, state):
        mu = state[0].get_mu()
        heuristic_value = math.sqrt((float(mu[0])-self.goal_region.centroid.x)**2 + (float(mu[1])-self.goal_region.centroid.y)**2) + max(0, self.goal_num_sample-state[1])
        return heuristic_value

    def execution_risk_heuristic(self, state):
        return 0


    def simulate_transition(self, state, best_action):
        state_distribution_set = self.state_transitions(state, best_action)
        states = []
        dist = []
        for i in state_distribution_set:
            states.append(i[0])
            dist.append(i[1])
        choices = range(len(states))
        choice = np.random.choice(choices, 1, p=dist)[0]
        return states[choice]

    def simulate_observation(self, state):
        obs_distribution_set = self.observations(state)
        obs = []
        dist = []
        for i in obs_distribution_set:
            obs.append(i[0])
            dist.append(i[1])

        choices = range(len(obs))
        choice = np.random.choice(choices, 1, p=dist)[0]
        return obs[choice]

    def compute_risk(self,mu, std, obs_vertices):
        Hull = ConvexHull(obs_vertices)
        Eq = Hull.equations
        Ao = Eq[:,0:2]
        bo = -Eq[:,2]
        R = []
        for i in range(bo.size):
            R.append(self.compute_risk_edge(Ao[i,:], bo[i], mu, std))

        return float(min(R))

    def compute_risk_edge(self,Ao, bo, mu, std):
        mu2 = Ao.dot(mu) - bo
        std2 = np.sqrt(Ao.dot(std*std).dot(Ao))
        return norm.cdf(0, mu2, std2) - norm.cdf(-np.inf, mu2, std2)


class GaussianState(object):

    def __init__(self, name=None, properties={}, mu=0, sigma=0):
        self.name = name
        self.properties = properties
        self.mu = mu
        self.sigma = sigma

    __hash__ = object.__hash__

    def get_mu(self):
        return self.mu

    def get_sigma(self):
        return self.sigma
    
    def __eq__(x, y):
        return isinstance(x, GaussianState) and isinstance(y, GaussianState) and (x.mu == y.mu).all() and (x.sigma == y.sigma).all()

    def __ne__(self, other):
        return not self == other
