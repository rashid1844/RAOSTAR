#!/usr/bin/env python

# author: Sungkweon Hong
# email: sk5050@mit.edu
# Grid model as simple model to test rao star

import math
import numpy as np
import time
from belief import BeliefState

class GRIDModel(object):

    def __init__(self, size=(3,3), constraint_states = [(1,1)], rewrd_states={(0,0):1}, prob_right_transition=0.9, prob_right_observation=0.9, DetermObs=False, goal=(4,4), step_cost=1, walls=[(1,1)]):

        # grid with size (x,y): width is x and height is y. index start from 0.
        # example of (3,3)
        # _____________
        # |0,2|1,2|2,2|
        # |___|___|___|
        # |0,1|1,1|2,1|
        # |___|___|___|
        # |0,0|1,0|2,0|
        # |___|___|___|
        #
        # observation is the min(x-goal.x, y-goal.y).
        self.walls = walls
        self.step_cost = step_cost
        self.DetermObs = DetermObs
        self.rewrd_states = rewrd_states
        self.prob_right_transition = prob_right_transition
        self.prob_right_observation = prob_right_observation
        self.size = size
        self.state_list = []
        self.goal = goal  # right upper corner.
        # self.goal = (9,4)
        self.obs_list = [0,1,2]

        # self.obs_list = [0,1,2,3,4,5,6,7,8]

        for i in range(size[0]):
            for j in range(size[1]):
                self.state_list.append((i,j))

        self.action_list = ["U","L","R","D"]
        # self.action_list = ["D","R"]


        self.constraint_states = constraint_states
        self.anytime_history = []
        self.anytime_update_time = []
        self.anytime_update_action = []
        self.anytime_update_risk = []
        self.anytime_policy = []
        self.percentage_history = []


        #self.optimization = 'maximize'
        self.optimization = 'minimize'

    def actions(self, state):
        return self.action_list

    def is_terminal(self, state):
        return state == self.goal

    def state_transitions(self, state, action):
        if action=="U":
            new_states_temp = [[(state[0]+1,state[1]), self.prob_right_transition],
                               [(state[0],state[1]+1), (1-self.prob_right_transition)/2],
                               [(state[0],state[1]-1), (1-self.prob_right_transition)/2]]

        elif action=="D":
            new_states_temp = [[(state[0]-1,state[1]), self.prob_right_transition],
                               [(state[0],state[1]+1), (1-self.prob_right_transition)/2],
                               [(state[0],state[1]-1), (1-self.prob_right_transition)/2]]

        elif action=="L":
            new_states_temp = [[(state[0],state[1]-1), self.prob_right_transition],
                               [(state[0]+1,state[1]), (1-self.prob_right_transition)/2],
                               [(state[0]-1,state[1]), (1-self.prob_right_transition)/2]]

        elif action=="R":
            new_states_temp = [[(state[0],state[1]+1), self.prob_right_transition],
                               [(state[0]+1,state[1]), (1-self.prob_right_transition)/2],
                               [(state[0]-1,state[1]), (1-self.prob_right_transition)/2]]

        prob_stay = 0
        new_states = []
        for new_state in new_states_temp:
            if new_state[0] not in self.state_list or new_state[0] in self.walls:
                prob_stay = prob_stay + new_state[1]
            else:
                new_states.append(new_state)

        if prob_stay > 0:
            new_states.append([state, prob_stay])

        return new_states

    def observations(self, state):
        if self.DetermObs:
            return [(state, 1.0)]
        else:
            
            right_obs = 0
            if state[0]==self.size[0]-1 or (state[0]+1,state[1]) in self.walls:  # up
                right_obs += 1
            if state[0]==0 or (state[0]-1,state[1]) in self.walls:  # down
                right_obs += 1
            if state[1]==0 or (state[0],state[1]-1) in self.walls:  # left
                right_obs += 1
            if state[1]==self.size[1]-1 or (state[0],state[1]+1) in self.walls:  # right
                right_obs += 1

            obs_dist = []

            for obs in self.obs_list:
                if obs==right_obs:
                    obs_dist.append((obs, self.prob_right_observation))
                else:
                    obs_dist.append((obs, (1-self.prob_right_observation)/(len(self.obs_list)-1)))

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
        if state in self.constraint_states:
            return 1.0
        else:
            return 0.0

    def values(self,state,action):  # TODO: make it based on a'=T(a,s)
        new_state = self.state_transitions(state, action)
        value = 0.
        for st, prob in new_state:
            value += self.rewrd_states.get(st, self.step_cost) * prob

        return value

    def heuristic(self, state):
        #return 0#math.sqrt((state[0]-self.goal[0])**2 + (state[1]-self.goal[1])**2)
        if state in self.rewrd_states.keys():
            return 0
        h = float('inf')
        for goal in self.rewrd_states.keys():
            # h = max(h, math.sqrt((state[0]-goal[0])**2 + (state[1]-goal[1])**2))
            h = min(h, (abs(state[0]-goal[0]) + abs(state[1]-goal[1]))*self.step_cost - 1)
            # break  # TODO: remove
        return h



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
