#!/usr/bin/env python


# Grid model as simple model to test rao star

import math
import numpy as np
import time
from belief import BeliefState

class NetworkModel(object):

    def __init__(self,  DetermObs=False):

        self.DetermObs = DetermObs

        self.state_list = [0,1,2,3,4,5,6]
        self.obs_list = [0,1]

        # self.obs_list = [0,1,2,3,4,5,6,7,8]


        self.action_list = ["U","S","R","B"]
        self.action_dict = {"U":0, "S":1, "R":2, "B":3}
        # self.action_list = ["D","R"]


        self.anytime_history = []
        self.anytime_update_time = []
        self.anytime_update_action = []
        self.anytime_update_risk = []
        self.anytime_policy = []
        self.percentage_history = []

        self.state_trans = np.array([[[0.5,0.3,0.1,0.1,0,0,0], [0.2,0.3,0.3,0.1,0.1,0,0], [0.1,0.1,0.3,0.3,0.1,0.1,0], [0,0.1,0.1,0.3,0.3,0.1,0.1],[0,0,0.1,0.1,0.3,0.3,0.2],[0,0,0,0.1,0.1,0.3,0.5],[0,0,0,0,0,0,1.],],
                            [[0.7,0.2,0.1,0.,0.,0.,0.],[0.3,0.4,0.2,0.1,0.,0.,0.],[0.1,0.2,0.4,0.2,0.1,0.,0.],[0.,0.1,0.2,0.4,0.2,0.1,0.],[0.,0.,0.1,0.2,0.4,0.2,0.1],[0.,0.,0.,0.1,0.2,0.4,0.3],[0.,0.,0.,0.,0.,0.,1.],],
                            [[0.8,0.1,0.1,0.,0.,0.,0.],[0.5,0.3,0.1,0.1,0.,0.,0.],[0.2,0.3,0.3,0.1,0.1,0.,0.],[0.1,0.1,0.3,0.3,0.1,0.1,0.],[0.1,0.,0.1,0.3,0.3,0.1,0.1],[0.,0.1,0.,0.1,0.3,0.3,0.2],[0.,0.,0.,0.,0.,0.,1.],],
                            [[1.,0.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.],[1.,0.,0.,0.,0.,0.,0.],],], np.float64)

        self.observation_list = np.array([[1,1,1,0.9,0.7,0.5,0],[0,0,0,0.1,0.3,0.5,1]], np.float64)

        #self.reward = np.array([[-20, 0, 20, 40, 60, 80, -20], [-20, 0, 20, 40, 60, 80, -20], [-20, 0, 20, 40, 60, 80, -20], [-40, -40, -40, -40, -40, -40, -40],])
        self.reward = np.array([[100,80,60,40,20,0,100],[100,80,60,40,20,0,100],[100,80,60,40,20,0,100],[120,120,120,120,120,120,120]], np.float64)
        #self.reward = np.array([[0,0,20,40,60,80,0],[0,0,20,40,60,80,0],[0,0,20,40,60,80,0],[0,0,0,0,0,0,0]], np.float64)

        self.optimization = 'minimize'

    def actions(self, state):
        return self.action_list

    def is_terminal(self, state):
        return False

    def state_transitions(self, state, action):
        new_states = []
        for i, prob in enumerate(self.state_trans[self.action_dict[action]][state]):
            new_states.append([i, prob])
        return new_states

    def observations(self, state):
            obs_dist = [(0, self.observation_list[0][state]), (1, self.observation_list[1][state])]
            return obs_dist

    # def observations(self, state):
    #     return [(state, 1.0)] # deterministic for now

    # def observations(self, state):
    #
    #     right_obs = state[0]
    #
    #     obs_dist = []
    #
    #     for obs in self.obs_list:
    #         if obs==right_obs:
    #             obs_dist.append((obs, self.prob_right_observation))
    #         else:
    #             obs_dist.append((obs, (1-self.prob_right_observation)/(len(self.obs_list)-1)))
    #
    #     obs_dist = [(state, 1.0)]
    #     return obs_dist



    # def observations(self, state):
    #
    #     if state[0]>= 0 and state[0]<=1 and state[1]>=0 and state[1]<=1:
    #         right_obs = 0
    #
    #     elif state[0]>= 2 and state[0]<=3 and state[1]>=0 and state[1]<=1:
    #         right_obs = 1
    #
    #     elif state[0]>= 4 and state[0]<=5 and state[1]>=0 and state[1]<=1:
    #         right_obs = 2
    #
    #     elif state[0]>= 0 and state[0]<=1 and state[1]>=2 and state[1]<=3:
    #         right_obs = 3
    #
    #     elif state[0]>= 2 and state[0]<=3 and state[1]>=2 and state[1]<=3:
    #         right_obs = 4
    #
    #     elif state[0]>= 4 and state[0]<=5 and state[1]>=2 and state[1]<=3:
    #         right_obs = 5
    #
    #     elif state[0]>= 0 and state[0]<=1 and state[1]>=4 and state[1]<=5:
    #         right_obs = 6
    #
    #     elif state[0]>= 2 and state[0]<=3 and state[1]>=4 and state[1]<=5:
    #         right_obs = 7
    #
    #     elif state[0]>= 4 and state[0]<=5 and state[1]>=4 and state[1]<=5:
    #         right_obs = 8
    #
    #
    #     obs_dist = []
    #
    #     for obs in self.obs_list:
    #         if obs==right_obs:
    #             obs_dist.append((obs, self.prob_right_observation))
    #         else:
    #             obs_dist.append((obs, (1-self.prob_right_observation)/(len(self.obs_list)-1)))
    #
    #     return obs_dist



    def state_risk(self, state):
        return 1. if state==6 else 0.

    def values(self,state,action):
        return self.reward[self.action_dict[action]][state]

    def heuristic(self, state):
        return 0#math.sqrt((state[0]-self.goal[0])**2 + (state[1]-self.goal[1])**2)

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
