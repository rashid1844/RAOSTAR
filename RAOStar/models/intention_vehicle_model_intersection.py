#!/usr/bin/env python

# author: Cyrus Huang
# email: huangxin@mit.edu
# intention-aware vehicle models with probabilistic flow tubes
#edited by Rashid Alyassi
import numpy as np
import copy
from pprint import pformat
import pickle
import matplotlib.pyplot as plt
import itertools
import IPython
import sys
import os
import math
import time
import csv
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PFT_DIR = os.path.join(WORKSPACE_DIR,'PFT')
sys.path.insert(0, PFT_DIR)

# get pft modules and pft data
import pft # need to get pft modules
PFT_DATA_DIR = os.path.join(PFT_DIR,'pft_data')

def calculate_permutations(model, state, actions, d):
    state_seq = [state]
    permutations = []
    for a in actions:
        action_seq = [a]
        new_seq = extend_sequence(
            model, state_seq, [a], actions, 0, d)
        if new_seq:
            if isinstance(new_seq, list):
                for i in new_seq:
                    permutations.append(i)
            else:
                permutations.append(new_seq)
    return permutations


def extend_sequence(model, state_seq, action_seq, actions, t, d):
    latest_action = action_seq[-1]
    latest_state = state_seq[-1]
    print('extend_sequence', model, state_seq, action_seq, t, d)

    if not (latest_action.preconditions(latest_state, model)):
        # Cannot use this sequence ending in lastest_action, fails
        # preconditions
        return None

    # time of next decision epoch is lastest action completed:
    new_t = t + latest_action.length_of_action

    if new_t > d:
        # Then we have an incompleted action sequence
        return {'states': state_seq, 'actions': action_seq, 'fraction': (d - t) / latest_action.length_of_action, 'status': 'INCOMPLETE'}
    else:
        new_agent_state = latest_action.effects(latest_state, model)
        new_multi_state = MultiState(new_t)
        new_multi_state.add_vehicle_state(new_agent_state)
        new_state_seq = list(state_seq)
        new_state_seq.append(new_multi_state)
        if new_t == d:
            print('perfect timing')
            # Perfect timing, synchronized action endorse
            return {'states': new_state_seq, 'actions': action_seq, 'fraction': 1, 'status': 'COMPLETE'}
        else:
            permutations = []
            for a in actions:
                new_action_seq = list(action_seq)
                new_action_seq.append(a)
                new_seq = extend_sequence(
                    model, new_state_seq, new_action_seq, actions, new_t, d)
                if new_seq:
                    if isinstance(new_seq, list):
                        for i in new_seq:
                            permutations.append(i)
                    else:
                        permutations.append(new_seq)
            return permutations


class VehicleState(object):                                                                           #POMDP 'S' state for single vehicle
    '''Object to represent a single vehicle state in the Geordi Model. Includes                   
    position, velocity, and additional attributes to describe a given state'''                        #state=(x,y,yaw) coordinates

    def __init__(self, name="Unnamed", state=None, previous_action=None, waiting_counter=1, turn_bool=False):
        self.name = name
        self.state = state
        self.previous_action = previous_action
        self.waiting_counter = waiting_counter
        self.turn_bool = turn_bool

    def __repr__(self):
        info = {'name': self.name, 'state': self.state, 'pre-action':self.previous_action}
        # return pformat(info)
        return 'VehicleState(name: {}, state: {}, previous_action: {}, Turn_bool: {})'.format(self.name,str(self.state),self.previous_action,self.turn_bool)

class MultiState(object):                                                                                #model of multiple cars, used in Geordi model "POMDP"
    '''Object to store composite state of the world in Geordi Model. Has a'''                            #list of all vehicles state object (list of VehicleState objects)
    '''dictionary that contains the 'vehicle-name': 'vehicle-state' for all the
    vehicles in a given multi-state. Also has utility functions like
    get_state(vehicle-name) and print_multistate()'''

    def __init__(self, timestep=-1):
        self.timestep = timestep
        self.multi_state = {}                                                                            #all states saved as a list and dictionary
        self.multi_state_list = []

    def add_vehicles_states_list(self, vehicle_states):
        for i in vehicle_states:
            self.add_vehicle_state(i)

    def add_vehicle_state(self, vehicle_state):
        if vehicle_state.name not in self.multi_state:
            self.multi_state[vehicle_state.name] = vehicle_state
            self.multi_state_list.append(vehicle_state)
        else:
            raise ValueError(
                "MultiState already has a state for " + str(vehicle_state.name))

    def get_vehicle_state_list(self):
        return self.multi_state_list

    def get_state(self, name):
        if name in self.multi_state:
            return self.multi_state[name]
        raise ValueError('MultiState does not have name: ' + str(name))

    def get_multistate_str(self):
        extended_state = self.multi_state
        extended_state['timestep'] = self.timestep
        return str(extended_state)

    def print_multistate(self):
        print(self.get_multistate_str())

    def __repr__(self):
        return 'MultiState({})'.format(self.get_multistate_str())


class GeordiModel(object):                                                                                      #The POMDP model of the problem
    '''First Python version of Geordi vehicles model, this one for intersections.'''                            #combines road model with all vehicle models
                                                                                                                #has starting point and ending point
    '''    Attributes:
        name (str): Name to id each vehicle.
        current_state (dictionary): Maps variables to values for current state
        of the vehicle. Starts with initial state and is used during execution.
        attr2 (:obj:`int`, optional): Description of `attr2`.
    '''

    def __init__(self, vehicle_models=[], road=None, goal_state=(0,0), shift=10):
        print('Made GeordiModel! 2.0')
        self.optimization = 'minimize'  # want to minimize the steps to goal                  used in RAOSTAR #TODO: minimize means each stop function "one sec delay" is cost of one
        self.shift = shift  # amount of shift (delay) for each horizon
        self.vehicle_models = {} #dictionery of vehicle models
        self.start_time = time.time()
        self.risk_time = 0.0
        self.risk_algo1 = 0.0
        self.risk_algo2 = 0.0
        self.risk_count = 0
        self.pft_list = {}
        self.get_all_pft()
        self.road_model = road
        self.controllable_vehicle = None                               #index of ego vehicle
        # Start with MultiState at timestep=0
        self.current_state = MultiState(0)
        self.goal_state = goal_state
        self.traj_list = {}
        self.traj_length = {}
        self.get_all_traj()
        # For all vehicle_models: add to MultiState and dict of vehicles
        for i in vehicle_models:
            self.current_state.add_vehicle_state(i.current_state)      #adds all vehicle states to a multistate
            self.vehicle_models[i.name] = i
            print(i)
            if i.controllable:
                self.controllable_vehicle = i
        print("GeordiModel MultiState: ",
              self.current_state.get_multistate_str())

        self.intersec_vehicle_dict = {vehicle_models[i].name: vehicle_models[i] for i in range(len(vehicle_models))}

        self.dict_goal_x = {'side_A_left_left': 31, 'side_A_left_forward': 43.5, 'side_A_right_forward': 43.5,
                            'side_A_right_right': 24, 'side_B_left_left': 28, 'side_B_left_forward': 15.5,
                            'side_B_right_forward': 15.5, 'side_B_right_right': 35.5,
                            'side_C_left_left': 44, 'side_C_left_forward': 28, 'side_C_right_forward': 24.5,
                            'side_C_right_right': 15.5, 'side_D_left_left': 15.5, 'side_D_left_forward': 31.5,
                            'side_D_right_forward': 35, 'side_D_right_right': 44}
        self.dict_goal_y = {'side_A_left_left': 74.5, 'side_A_left_forward': 91.5, 'side_A_right_forward': 95,
                            'side_A_right_right': 108, 'side_B_left_left': 104, 'side_B_left_forward': 88.5,
                            'side_B_right_forward': 85, 'side_B_right_right': 74.5,
                            'side_C_left_left': 92, 'side_C_left_forward': 105, 'side_C_right_forward': 105,
                            'side_C_right_right': 84, 'side_D_left_left': 88, 'side_D_left_forward': 74,
                            'side_D_right_forward': 74, 'side_D_right_right': 96}

        # index of traj at goal (NOT USED)
        self.dict_goal_index = {'side_A_left_left': 250, 'side_A_left_forward': 230, 'side_A_right_forward': 260,
                       'side_A_right_right': 320, 'side_B_left_left': 220, 'side_B_left_forward': 200,
                       'side_B_right_forward': 220, 'side_B_right_right': 220, 'side_C_left_left': 340,
                       'side_C_left_forward': 230, 'side_C_right_forward': 240, 'side_C_right_right': 240,
                       'side_D_left_left': 240, 'side_D_left_forward': 220, 'side_D_right_forward': 250,
                       'side_D_right_right': 220}

        self.save_risk = True
        if self.save_risk:
            self.save_counter = 0
            self.action_map_binary = {'side_A_left_left': 32768, 'side_A_left_forward': 16384,
                                      'side_A_right_forward': 8192,
                                      'side_A_right_right': 4096, 'side_B_left_left': 2048, 'side_B_left_forward': 1024,
                                      'side_B_right_forward': 512, 'side_B_right_right': 256,
                                      'side_C_left_left': 128, 'side_C_left_forward': 64, 'side_C_right_forward': 32,
                                      'side_C_right_right': 16, 'side_D_left_left': 8, 'side_D_left_forward': 4,
                                      'side_D_right_forward': 2, 'side_D_right_right': 1, 'stop': 0}
            with open('one_speed_risk.csv', 'w', newline="") as file:
                writer = csv.writer(file)
                writer.writerow(['binary_id', 'risk'])


    def get_all_traj(self):  # loads ego car predifined trajectory     [ [timestamp, pos_x, pos_y, pos_z, vel, yaw] ....]
        traj_names = ["Trajectory/intersection_controlled_side_A_left_left.csv", "Trajectory/intersection_controlled_side_A_left_forward.csv",
                      "Trajectory/intersection_controlled_side_A_right_forward.csv",
                      "Trajectory/intersection_controlled_side_A_right_right.csv",
                      "Trajectory/intersection_controlled_side_B_left_left.csv", "Trajectory/intersection_controlled_side_B_left_forward.csv",
                      "Trajectory/intersection_controlled_side_B_right_forward.csv",
                      "Trajectory/intersection_controlled_side_B_right_right.csv",
                      "Trajectory/intersection_controlled_side_C_left_left.csv", "Trajectory/intersection_controlled_side_C_left_forward.csv",
                      "Trajectory/intersection_controlled_side_C_right_forward.csv",
                      "Trajectory/intersection_controlled_side_C_right_right.csv",
                      "Trajectory/intersection_controlled_side_D_left_left.csv", "Trajectory/intersection_controlled_side_D_left_forward.csv",
                      "Trajectory/intersection_controlled_side_D_right_forward.csv",
                      "Trajectory/intersection_controlled_side_D_right_right.csv", ]
        for file in traj_names:
            self.traj_list[file[35:-4]] = np.genfromtxt(file, delimiter=',')  # I have pre-removed the trajectory part before the car starts moving
            self.traj_list[file[35:-4]] = self.traj_list[file[35:-4]][1:]  # to remove the header
            self.traj_length[file[35:-4]] = len(self.traj_list[file[35:-4]])  # save the length of the traj
            # traj_list['side_D_right_right'] (sample usage)




    def add_vehicle_model(self, vehicle_model):  # adds a vehicle model to the list
        new_name = vehicle_model.name
        if new_name not in self.vehicle_models:
            # If not saved in model, add to vehicle list and MultiState
            self.vehicle_models[new_name] = vehicle_model
            self.current_state.add_vehicle_state(self.current_state)                       #i is a typo "self"
            # If we are adding the controllable vehicle after init GeordiModel
            if vehicle_model.controllable:
                if self.controllable_vehicle:
                    raise ValueError(
                        'GeordiModel already has one controllable vehicle')
                self.controllable_vehicle = vehicle_model
        # Already in vehicle list? Raise an Error
        else:
            raise ValueError(
                'GeordiModel already has VehicleModel: ' + new_name)

    def get_vehicle_model(self, name):                                 #return vehicle model
        if name in self.vehicle_models:
            return self.vehicle_models[name]

    def get_vehicle_state(self, name):                               #return vehicle state
        return self.get_vehicle_mode(name).name


    # to check if prereq agent has moved yet or not (return True is there is prerequests)
    def prerequisite(self, multi_state, agent):
        # No car in-front
        if self.intersec_vehicle_dict[agent].prerequisite is None:
            return False
        pre_agent = self.intersec_vehicle_dict[agent].prerequisite

        #if '9' in pre_agent:
        #    IPython.embed(header='prerequiste 9 and 11')

        # Car in-front left intersection
        if pre_agent not in multi_state.multi_state:
            return False

        # Car in-front didn't move yet
        if not multi_state.multi_state[pre_agent].previous_action or 'stop' in multi_state.multi_state[pre_agent].previous_action.name:
            return True

        # Car in-front is in the intersection
        if 'side_' in multi_state.multi_state[pre_agent].previous_action.name:
            idx = -1
            # for cars in row (120 iterations to move forward)
            row_shift = int(self.shift * 3 // 2) if 'fast' in multi_state.multi_state[pre_agent].previous_action.name else self.shift
            for index, pos in enumerate(self.traj_list[multi_state.multi_state[pre_agent].previous_action.name.replace('_fast','')]):
                if abs(pos[1] - multi_state.multi_state[pre_agent].state['x']) < 0.2 and abs(pos[2] - multi_state.multi_state[pre_agent].state['y']) < 0.2 and\
                        abs(pos[5] - multi_state.multi_state[pre_agent].state['yaw']) < 0.2:  # and index % shift == 0
                    idx = index
                    break
            if idx >= row_shift-10:  # TODO change if changed in main function (after n iterations add dependent car)
                return False
            else:
                return True




    def get_available_actions(self, multi_state):       #list of 'possible' actions for ego vehicle   (checks precondition function for action which uses geordi model
        '''                                                    
        Return set of controllable actions available from given multi-state
        This list of actions is just pruned by the road model itself, not nearby
        vehicles. We are not enforcing safe driving through this filter.
        '''                                                  # TODO: change it to show list of all posible actions based on state preaction (DONE)

        # exp of multi_state: MultiState({'Ego': VehicleState(name: Ego, state: {'x': 31.5, 'y': 105, 'yaw': 270}, previous_action: None), 'Agent0': VehicleState(name: Agent0, state: {'x': 41.79261016845703, 'y': 88.6486587524414, 'yaw': 176.32565307617188}, previous_action: None), 'Agent1': VehicleState(name: Agent1, state: {'x': 24.5, 'y': 72.5, 'yaw': 90}, previous_action: None), 'timestep': 0})
        # multi_state.multi_state.keys() --->>> dict_keys(['Ego', 'Agent0', 'Agent1', 'timestep'])
        # print(multi_state.multi_state['Agent0']) --->>>  VehicleState(name: Agent0, state: {'x': 41.79261016845703, 'y': 88.6486587524414, 'yaw': 176.32565307617188}, previous_action: None)
        # print(multi_state.multi_state['Agent0'].previous_action) --->>> None

        # print(self.intersec_vehicle) --->>>  [VehicleModel(VehicleState(name: Ego, state: {'x': 31.5, 'y': 105, 'yaw': 270}, previous_action: None)), VehicleModel(VehicleState(name: Agent0, state: {'x': 41.79261016845703, 'y': 88.6486587524414, 'yaw': 176.32565307617188}, previous_action: None)), VehicleModel(VehicleState(name: Agent1, state: {'x': 24.5, 'y': 72.5, 'yaw': 90}, previous_action: None))]

        # print(self.intersec_vehicle_dict) --->>> {'Ego': VehicleModel(VehicleState(name: Ego, state: {'x': 31.5, 'y': 105, 'yaw': 270}, previous_action: None)), 'Agent0': VehicleModel(VehicleState(name: Agent0, state: {'x': 41.79261016845703, 'y': 88.6486587524414, 'yaw': 176.32565307617188}, previous_action: None)), 'Agent1': VehicleModel(VehicleState(name: Agent1, state: {'x': 24.5, 'y': 72.5, 'yaw': 90}, previous_action: None))}
        # print(self.intersec_vehicle_dict['Agent0'].get_actions()) --->>> [ActionModel(side_B_left_left)]   Note: output is list of ActionModel objects


        action_list = []  # [ [agent_1 actions] , [agent_2 actions] ....]
        count = 0  # number of agents
        for agent in multi_state.multi_state.keys(): #to check if agent have moved yet or not (stop all lower case)
            if agent != 'timestep':
                if not multi_state.multi_state[agent].turn_bool:# multi_state.multi_state[agent].previous_action == None or 'stop' in multi_state.multi_state[agent].previous_action.name:
                    if not self.prerequisite(multi_state, agent):
                        action_list.append(self.intersec_vehicle_dict[agent].get_actions())
                        # if there is prerequests, it wont be considerd in the action list
                        count += 1
                    else:
                        # stop actions is added first so it's always index zero
                        act_list = self.intersec_vehicle_dict[agent].get_actions()
                        action_list.append([act_list[0]])

        full_action_list = list(itertools.product(*action_list))
        #IPython.embed(header='get_action')

        # Takes percentage of all actions randomly
        count = max(count, 1)  # to avoid count = 0
        count += 5 if count > 6 else 0
        half_list = full_action_list[::(count+1)//2]

        return half_list  # output list of all possible action combinations (each action is an action model) actions are sorted based on action name Agent0 1 2 ....



    '''
    ##  API required by RAO*  ##
        self.A = model.actions
        self.T = model.state_transitions
        self.O = model.observations
        self.V = model.values
        self.h = model.heuristic
        self.r = model.state_risk
        self.e = model.execution_risk_heuristic
        self.term = model.is_terminal
    '''

    def observations(self, state):              #returns any observation with probability 1
        return [(state, 1.0)]

    def actions(self, state):
        '''RAO* API - return list of available_actions in given multi-state state'''
        return self.get_available_actions(state)


    def state_transitions(self, state, action):

        # state: state.multi_state[agent_name].state = {'x': 1, 'y': 2, 'yaw': 3}
        # action: [ActionModel1, ActionModel1, ActionModel1, ActionModel1 ]

        # exp of multi_state: MultiState({'Ego': VehicleState(name: Ego, state: {'x': 31.5, 'y': 105, 'yaw': 270}, previous_action: None), 'Agent0': VehicleState(name: Agent0, state: {'x': 41.79261016845703, 'y': 88.6486587524414, 'yaw': 176.32565307617188}, previous_action: None), 'Agent1': VehicleState(name: Agent1, state: {'x': 24.5, 'y': 72.5, 'yaw': 90}, previous_action: None), 'timestep': 0})
        # multi_state.multi_state.keys() --->>> dict_keys(['Ego', 'Agent0', 'Agent1', 'timestep'])
        # print(multi_state.multi_state['Agent0']) --->>>  VehicleState(name: Agent0, state: {'x': 41.79261016845703, 'y': 88.6486587524414, 'yaw': 176.32565307617188}, previous_action: None)
        # print(multi_state.multi_state['Agent0'].previous_action) --->>> None

        '''
        First checks agents that are still moving, if they didn't reach goal, shift them by self.shift times
        next for stationary items, if action is move, then place them at state (initial state + self.shift)
        finally just copy the states that will remain stationary
        '''
        #import IPython; IPython.embed(header='state_trans')
        try:
            #new_state = MultiState(timestep=state.multi_state['timestep'] + self.shift)
            new_state = MultiState(timestep=state.timestep + self.shift)
        except:
            print(state.multi_state['timestep'])
            pass

        stationary_agent = []

        # for moving agents
        for agent in state.multi_state.keys():
            if agent != 'timestep':
                # for stationary agents
                if not state.multi_state[agent].turn_bool:  # 'stop' in state.multi_state[agent].previous_action.name:
                    stationary_agent.append(agent)

                else:  # 'side_' in state.multi_state[agent].previous_action.name: #if previous action was turn
                    idx = -1
                    #goal_bool = False

                    # checks is you reach goal by goal position
                    #if abs(self.dict_goal_x[state.multi_state[agent].previous_action.name] - state.multi_state[agent].state['x']) < 3 and abs(
                    #        self.dict_goal_y[state.multi_state[agent].previous_action.name] - state.multi_state[agent].state['y']) < 3:
                    #    goal_bool = True
                    #    print(agent, 'has reached goal###########################')

                    shift = int(self.shift * 3 // 2) if 'fast' in state.multi_state[agent].previous_action.name else self.shift

                    #  checks if you reach goal by your index >= pft size
                    for index, pos in enumerate(self.traj_list[state.multi_state[agent].previous_action.name.replace('_fast', '')]):
                        if index % shift == 0 and abs(pos[1]-state.multi_state[agent].state['x']) < 0.2 and abs(pos[2]-state.multi_state[agent].state['y']) < 0.2 and abs(pos[5]-state.multi_state[agent].state['yaw']) < 0.2:
                            idx = index
                            break

                    if idx == -1:
                        print('dix error transtionn fuction #############################################')

                    # quick fix for big shift values skipping goal (10 becuase pft was recorded at each 10 step)
                    #if idx > self.dict_goal_index[state.multi_state[agent].previous_action.name]:
                    # idx + shift: because we check if after taking the action he will go past goal

                    if np.round((idx+shift)/10) >= self.pft_list[state.multi_state[agent].previous_action.name.replace('_fast', '') + "_pft.pkl"].l:
                        continue
                    else:  # if goal is reached agent wont be added to state (agent is kept in state after action turn to compute risk)
                        new_state.multi_state[agent] = VehicleState(name=agent, state={'x': self.traj_list[state.multi_state[agent].previous_action.name.replace('_fast', '')][idx+shift][1],
                                                                                       'y': self.traj_list[state.multi_state[agent].previous_action.name.replace('_fast', '')][idx+shift][2],
                                                                                       'yaw': self.traj_list[state.multi_state[agent].previous_action.name.replace('_fast', '')][idx+shift][5]},
                                                                    previous_action=state.multi_state[agent].previous_action, waiting_counter=state.multi_state[agent].waiting_counter+1, turn_bool=True)



        # for stationary agents
        for index, agent in enumerate(stationary_agent):
            #agents just stated moving
            #try:
            if 'side' in action[index].name:
                shift = int(self.shift*3//2) if 'fast' in action[index].name else self.shift
                new_state.multi_state[agent] = VehicleState(name=agent, state={
                    'x': self.traj_list[action[index].name.replace('_fast', '')][shift][1],
                    'y': self.traj_list[action[index].name.replace('_fast', '')][shift][2],
                    'yaw': self.traj_list[action[index].name.replace('_fast', '')][shift][5]},
                                                            previous_action=action[index], turn_bool=True)
            else:
                new_state.multi_state[agent] = copy.copy(state.multi_state[agent])
                new_state.multi_state[agent].waiting_counter += 1
                # agents that will remain stationary

            #except :
            #    print("Unexpected error:", sys.exc_info()[0])
            #    IPython.embed(header='state_transitions tuple index out of range')

        return [[new_state, 1.0]]  # return list of states with probability


    def goal_function(self, state):          #checks if ego vehicle state equals goal state
        goal_state_keys = list(self.goal_state.keys())

        if 'x' in goal_state_keys and 'y' in goal_state_keys:
            if state.get_state('Ego').state['x'] == self.goal_state['x'] \
               and state.get_state('Ego').state['y'] == self.goal_state['y']:
                return True
        elif 'y' in goal_state_keys:
            if state.get_state('Ego').state['y'] == self.goal_state['y']:
                return True
        elif 'x' in goal_state_keys:
            if state.get_state('Ego').state['x'] >= self.goal_state['x']:
                return True


    # def goal_function(self, state):
    #     if state.get_state('Ego').state['x'] > 10:
    #         return True

    def is_terminal(self, state):                           #checks if ego vehicle state equals goal state
        '''RAO* API - return True if this state is terminal, either planning
        horizon edge, goal state, or terminal failure'''
        # if self.iterative_deepening:
        # if state.depth == self.planning_horizon:
        # return True

        # exp of multi_state: MultiState({'Ego': VehicleState(name: Ego, state: {'x': 31.5, 'y': 105, 'yaw': 270}, previous_action: None), 'Agent0': VehicleState(name: Agent0, state: {'x': 41.79261016845703, 'y': 88.6486587524414, 'yaw': 176.32565307617188}, previous_action: None), 'Agent1': VehicleState(name: Agent1, state: {'x': 24.5, 'y': 72.5, 'yaw': 90}, previous_action: None), 'timestep': 0})
        # multi_state.multi_state.keys() --->>> dict_keys(['Ego', 'Agent0', 'Agent1', 'timestep'])
        # print(multi_state.multi_state['Agent0']) --->>>  VehicleState(name: Agent0, state: {'x': 41.79261016845703, 'y': 88.6486587524414, 'yaw': 176.32565307617188}, previous_action: None)
        # print(multi_state.multi_state['Agent0'].previous_action) --->>> None

        for agent in state.multi_state.keys():
            if agent != 'timestep':
                if not state.multi_state[agent].previous_action or 'stop' in state.multi_state[agent].previous_action.name:
                    return False
        return True
        # TODO change terminal to all cars being out of state list(Done)


    def load_pft(self, action_name):                           #returns pft of a given action
        action_name = ''.join([i for i in action_name if not i.isdigit()])
        pft_path = os.path.join(PFT_DATA_DIR, '%s_pft_short.pkl' % (action_name))
        with open(pft_path, 'rb') as f_snap:
            pft = pickle.load(f_snap)
            # print(action_name, pft.pos_mu)
        return pft

    def get_all_pft(self):
        pft_names=["pfts/A_left_left_pft.pkl", "pfts/A_left_forward_pft.pkl", "pfts/A_right_forward_pft.pkl", "pfts/A_right_right_pft.pkl",
                   "pfts/B_left_left_pft.pkl", "pfts/B_left_forward_pft.pkl", "pfts/B_right_forward_pft.pkl", "pfts/B_right_right_pft.pkl",
                   "pfts/C_left_left_pft.pkl", "pfts/C_left_forward_pft.pkl", "pfts/C_right_forward_pft.pkl", "pfts/C_right_right_pft.pkl",
                   "pfts/D_left_left_pft.pkl", "pfts/D_left_forward_pft.pkl", "pfts/D_right_forward_pft.pkl", "pfts/D_right_right_pft.pkl"]
        for name in pft_names:
            with open(name, 'rb') as f_snap:
                self.pft_list['side_' + name[5:]] = pickle.load(f_snap)
                # name[5:] to remove foldername 'pfts/'


    def state_risk(self, state):                            # TODO: create your own loop and actions and use this function(DONE)
        '''RAO* API - return 1.0 if multi-state breaks constraint'''
        # Compute risk as a probability of colliding with other vehicles
        # using probabilistic flow tubes representing vehicle actions
        temp_time = time.time()
        multi_state_keys = state.multi_state.keys()
        agent_names = list(filter(lambda x: x.startswith(('Agent')),
                                  multi_state_keys))  # all agents have name Agent in the beginning (better way, take all models with ego=False)
        '''
        p_collision = np.array([0.0])
        for agent1 in agent_names:
            for agent2 in agent_names:
                if agent1 == agent2:
                    continue

                agent1_previous_action = state.get_state(agent1).previous_action
                agent2_previous_action = state.get_state(agent2).previous_action

                if not agent1_previous_action or not agent2_previous_action:
                    p_collision = np.append(p_collision, 0.0)
                    continue

                if 'stop' in agent1_previous_action.name or 'stop' in agent2_previous_action.name:
                    p_collision =np.append(p_collision, 0.0)
                    continue

                # if one has prerequest so it wont enter the intersection
                if self.prerequisite(state,agent1) or self.prerequisite(state,agent2):
                    p_collision =np.append(p_collision, 0.0)
                    continue

                agent1_state = (self.dict_goal_x[state.multi_state[agent1].previous_action.name],
                                self.dict_goal_y[state.multi_state[agent1].previous_action.name])

                agent2_state = (self.dict_goal_x[state.multi_state[agent2].previous_action.name],
                                self.dict_goal_y[state.multi_state[agent2].previous_action.name])

                agent1_index = -1  # need to divide index by 10, to make it reassemble pft
                agent2_index = -1
                for index, pos in enumerate(self.traj_list[state.multi_state[agent1].previous_action.name]):
                    if abs(pos[1]-state.multi_state[agent1].state['x']) < 1 and abs(pos[2]-state.multi_state[agent1].state['y']) < 1 and abs(pos[5]-state.multi_state[agent1].state['yaw']) < 1:
                        agent1_index = index
                        break

                for index, pos in enumerate(self.traj_list[state.multi_state[agent2].previous_action.name]):
                    if abs(pos[1]-state.multi_state[agent2].state['x']) < 1 and abs(pos[2]-state.multi_state[agent2].state['y']) < 1 and abs(pos[5]-state.multi_state[agent2].state['yaw']) < 1:
                        agent2_index = index
                        break

                collision_result = self.compute_pairwise_action_risk(agent1_state, agent1_index//10, agent1_previous_action,
                                                                     agent2_state, agent2_index//10, agent2_previous_action)
                p_collision = np.append(p_collision, collision_result)

        risk = 1.0 - np.prod((1.0 - p_collision))  # risk= 1 - sum prob of no collision
        '''
        risk2 = self.compute_multi_risk(state, agent_names)
        self.risk_time += time.time() - temp_time

        #used for stats
        self.risk_algo1 += 0.0
        self.risk_algo2 += risk2
        self.risk_count += 1

        return risk2


    #computes risk of all vehicles in one loop
    def compute_multi_risk(self, state, agent_names, debug=False):
        safe_dist = 7
        n_samples = 10
        #previous_action = []
        agent_start = []
        agent_pft_list = []
        agent_index_list = []
        agent_action_list = []
        agent_action_fast = []
        agent_names_cp = agent_names.copy()
        starting_agents = []


        for agent in agent_names:
            agent_previous_action = state.get_state(agent).previous_action

            if not agent_previous_action or 'stop' in agent_previous_action.name or self.prerequisite(state, agent):
                agent_names_cp.remove(agent)
            else:
                #previous_action.append(agent_previous_action)

                agent_state = (self.dict_goal_x[state.multi_state[agent].previous_action.name.replace('_fast', '')],
                                self.dict_goal_y[state.multi_state[agent].previous_action.name.replace('_fast', '')])

                agent_pft = self.pft_list[agent_previous_action.name.replace('_fast', '') + "_pft.pkl"]
                agent_pft_list.append(agent_pft)
                agent_action_list.append(agent_previous_action.name)
                agent_action_fast.append(True if 'fast' in agent_previous_action.name else False)
                agent_pft_mu = agent_pft.pos_mu

                # TODO: could be sped up by adding index into the state values, and state transition
                agent_index = -1
                for index, pos in enumerate(self.traj_list[state.multi_state[agent].previous_action.name.replace('_fast', '')]):
                    if abs(pos[1]-state.multi_state[agent].state['x']) < 1 and abs(pos[2]-state.multi_state[agent].state['y']) < 1 and abs(pos[5]-state.multi_state[agent].state['yaw']) < 1:
                        agent_index = int(np.round(index//10))  # rounding is better than floor //
                        break

                if agent_index >= agent_pft.l:  # if agent index is bigger the pft, it changes it to the last position
                    agent_index = agent_pft.l - 1

                if agent_index == 0:
                    starting_agents.append(agent)

                agent_index_list.append(agent_index)
                agent_start.append((agent_state[0] - (agent_pft_mu[-1][0] - agent_pft_mu[agent_index][0]), agent_state[1] - (agent_pft_mu[-1][1] - agent_pft_mu[agent_index][1])))

        agent_names = agent_names_cp.copy()
        agent_len = len(agent_names)

        if self.save_risk and state.timestep == 0:
            binary_id = sum(self.action_map_binary[state.get_state(agent).previous_action] for agent in starting_agents)
        else:
            binary_id = -1

        if agent_len != 0:
            max_length = max(pft.l for pft in agent_pft_list)
            #collision_result = np.array([0.0] * max_length)
            p_collision = np.array([0.0] * (agent_len ** 2)).reshape(agent_len, agent_len)
            risk = 0
            for index in range(max_length):  # number of steps

                for a1 in range(agent_len - 1):

                    index_a1 = index*3//2 if agent_action_fast[a1] else index  # for fast action
                    if index_a1 + agent_index_list[a1] < agent_pft_list[a1].l:
                        # mu = index_pos - curent_pos + start_point
                        agent_mu1 = (agent_pft_list[a1].pos_mu[index_a1 + agent_index_list[a1]][0] - agent_pft_list[a1].pos_mu[agent_index_list[a1]][0] + agent_start[a1][0],
                                     agent_pft_list[a1].pos_mu[index_a1 + agent_index_list[a1]][1] - agent_pft_list[a1].pos_mu[agent_index_list[a1]][1] + agent_start[a1][1])
                        #agent_sigma1 = agent_pft_list[a1].pos_sigma[index_a1 + agent_index_list[a1]]
                    else:
                        continue

                    for a2 in range(a1+1, agent_len):

                        if agent_action_list[a1][:6] == agent_action_list[a2][:6]:  # if both cars are from same side, then risk=0
                            continue

                        if 'forward' in agent_action_list[a1] and 'forward' in agent_action_list[a2]:  # if both cars are from same side, then risk=0
                            temp = agent_action_list[a1] + agent_action_list[a2]
                            if ('side_A' in temp and 'side_B' in temp) or ('side_C' in temp and 'side_D' in temp):
                                continue

                        cp = 0.0
                        index_a2 = index*3//2 if agent_action_fast[a2] else index  # for fast action
                        if index_a2 + agent_index_list[a2] < agent_pft_list[a2].l:
                            agent_mu2 = (agent_pft_list[a2].pos_mu[index_a2 + agent_index_list[a2]][0] - agent_pft_list[a2].pos_mu[agent_index_list[a2]][0] + agent_start[a2][0],
                                         agent_pft_list[a2].pos_mu[index_a2 + agent_index_list[a2]][1] - agent_pft_list[a2].pos_mu[agent_index_list[a2]][1] + agent_start[a2][1])
                            #agent_sigma2 = agent_pft_list[a2].pos_sigma[index_a2 + agent_index_list[a2]]
                        else:
                            continue

                        safe_dist_t = int(safe_dist)
                        safe_dist_t += 3 if agent_action_fast[a1] else 0
                        safe_dist_t += 3 if agent_action_fast[a2] else 0

                        dist = (agent_mu1[0] - agent_mu2[0]) ** 2 + (agent_mu1[1] - agent_mu2[1]) ** 2

                        if dist < safe_dist_t ** 2:
                            '''
                            agent1_pos = np.random.multivariate_normal(agent_mu1, agent_sigma1, n_samples)
                            agent2_pos = np.random.multivariate_normal(agent_mu2, agent_sigma2, n_samples)
                            collision_cnt = 0

                            for p in range(n_samples):
                                agent1_p = agent1_pos[p]

                                for q in range(n_samples):
                                    agent2_p = agent2_pos[q]
                                    dist_pos = (agent1_p[0] - agent2_p[0]) ** 2 + (agent1_p[1] - agent2_p[1]) ** 2

                                    if dist_pos < safe_dist_t ** 2:
                                        collision_cnt += 1
                                        break

                            cp = 1.0 * collision_cnt / n_samples
                            '''
                            risk = 1
                            break

                        #p_collision[a1][a2] = max(cp, p_collision[a1][a2])

                #collision_result[index] = 1.0 - np.prod(1.0 - p_collision.flatten())

            #remove all zeros
            #p_collision = p_collision[p_collision != 0.0]

            #risk = 1.0 - np.prod(1.0 - p_collision.flatten())
            #risk = max(p_collision.flatten())  # TODO: pick one

            #if agent_len==8: #debug:
            #    IPython.embed(header='multirisk')

            if self.save_risk and binary_id != -1:
                self.save_counter += 1
                print('save counter:', self.save_counter)
                with open('one_speed_risk.csv', 'a', newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([binary_id, risk])

            return risk
        else:
            return 0.0



    # NOT USED
    def compute_pairwise_action_risk(self, ego_state, ego_index, ego_previous_action, agent_state, agent_index, agent_previous_action):
        verbose = False              # NOTE: state is future positon after action is taken, index is which step are at pft, start is were pft starts
        visualize = False                                             # computes risk
        safe_dist = 8                                               # state: current x,y location
        n_samples = 10                                                 # previous action: the selected action tobe taken #  TODO: reset
        # get ego and agent pfts                                        # index: current position with pft

        ego_pft = self.pft_list[ego_previous_action.name.replace('_fast', '') + "_pft.pkl"]
        agent_pft = self.pft_list[agent_previous_action.name.replace('_fast', '') + "_pft.pkl"]

        # print('Ego info:', ego_state, ego_previous_action, ego_pft.l)
        # print('Agent info:', agent_state,agent_previous_action, agent_pft.l)

        #ego_pft.l = min(ego_pft.l, agent_pft.l)
        #agent_pft.l = min(ego_pft.l, agent_pft.l)

        # assume pft's have equal lengths
        #assert ego_pft.l == agent_pft.l, '{} - {}, {} - {}'.format(ego_previous_action, ego_pft.l, agent_previous_action, agent_pft.l)  #checks if pfts are equal length
        ego_pft_mu = ego_pft.pos_mu
        ego_pft_sigma = ego_pft.pos_sigma
        agent_pft_mu = agent_pft.pos_mu
        agent_pft_sigma = agent_pft.pos_sigma

        # a quick hack
        # TODO: fix this
        if agent_index >= len(agent_pft_mu):        #if agent index is bigger the pft, it changes it to the last position
            agent_index = len(agent_pft_mu) - 1

        #import IPython; IPython.embed(header='pft')

        #if agent_index < 0:   #when agent hasn't moved yet
        #    ego_index = min(-agent_index, ego_pft.l) #adds the diffrence to ego
        #    agent_index = 0

        # TODO: if index > pft.l return risk = 0

        # compute starting position of each vehicle given their actions
        # based on state"end" - (last point - current index)
        ego_start = (ego_state[0] - (ego_pft_mu[-1][0] - ego_pft_mu[ego_index][0]),
                     ego_state[1] - (ego_pft_mu[-1][1] - ego_pft_mu[ego_index][1]))
        agent_start = (agent_state[0] - (agent_pft_mu[-1][0] - agent_pft_mu[agent_index][0]),
                       agent_state[1] - (agent_pft_mu[-1][1] - agent_pft_mu[agent_index][1]))


        # print('ego:', ego_start, ego_state)
        # print('agent:', agent_start, agent_state, agent_index)

        if visualize:
            plt.clf()
            plt.plot(ego_state[0], ego_state[1], 'rx')
            plt.plot(agent_state[0], agent_state[1], 'bx')
            plt.plot(ego_start[0], ego_start[1], 'ro')
            plt.plot(agent_start[0], agent_start[1], 'bo')
            print('agent_index',agent_index, agent_pft.l, agent_state, agent_start)

        p_collision = np.array([0.0]*ego_pft.l)
        for i in range(0, ego_pft.l, 1):  #TODO: remove (checks every 6 points)
            # compute shifted positions
            if i + ego_index < ego_pft.l:
                ego_mu = (ego_pft_mu[i + ego_index][0] - ego_pft_mu[ego_index][0] + ego_start[0],
                          ego_pft_mu[i + ego_index][1] - ego_pft_mu[ego_index][1] + ego_start[1])
                ego_sigma = ego_pft_sigma[i + ego_index]
            else:
                continue

            if i + agent_index < agent_pft.l:
                agent_mu = (agent_pft_mu[i + agent_index][0] - agent_pft_mu[agent_index][0] + agent_start[0],
                            agent_pft_mu[i + agent_index][1] - agent_pft_mu[agent_index][1] + agent_start[1])
                agent_sigma = agent_pft_sigma[i + agent_index]
            else:
                continue

            if visualize:
                plt.plot(ego_mu[0],ego_mu[1],'r.')
                plt.plot(agent_mu[0],agent_mu[1],'b.')

            # compute distance between pairwise locations
            dist = (ego_mu[0] - agent_mu[0]) ** 2 + (ego_mu[1] - agent_mu[1]) ** 2

            #Car length and width of tesla model 3
            L_1 = 4.0
            W_1 = 1.5
            L_2 = 4.0
            W_2 = 1.5
            max_dist = math.sqrt((L_1/2)**2 + (W_1/2)**2) + math.sqrt((L_2/2)**2 + (W_2/2)**2)
            min_dist = W_1/2 + W_2/2

            if dist < safe_dist ** 2:
                if verbose:
                    print('Vehicle too close. Checking probability of collision...')
                    print('Step {}, distance {:.2f}, ego_mu: {}, agent_mu: {}'.format(i, dist, ego_mu, agent_mu))

                ego_pos = np.random.multivariate_normal(ego_mu, ego_sigma, n_samples)
                agent_pos = np.random.multivariate_normal(agent_mu, agent_sigma, n_samples)
                collision_cnt = 0

                for p in range(n_samples):
                    ego_p = ego_pos[p]

                    for q in range(n_samples):
                        agent_p = agent_pos[q]
                        dist_pos = (ego_p[0] - agent_p[0]) ** 2 + (ego_p[1] - agent_p[1]) ** 2
                        '''
                        if dist_pos > max_dist:
                            break
                        if dist_pos <= min_dist:
                            collision_cnt += 1
                            break

                        theta_d = math.tanh((ego_p[0] - agent_p[0])/(ego_p[1] - agent_p[1]))
                        theta_1 = self.normalize_theta(ego_pft.ori_mu[i] - theta_d)
                        theta_2 = self.normalize_theta(agent_pft.ori_mu[i] - theta_d)  # TODO: add pft with theta variance

                        step_1 = math.tanh(W_1/L_1)
                        step_2 = math.tanh(W_2/L_2)

                        if theta_1 <= step_1:
                            d_1 = (L_1/2)/math.cos(theta_1)
                        else:
                            d_1 = (W_1/2)/math.sin(theta_1)

                        if theta_2 <= step_2:
                            d_2 = (L_2/2)/math.cos(theta_2)
                        else:
                            d_2 = (W_2/2)/math.sin(theta_2)

                        if d_1 + d_2 >= dist_pos:
                            collision_cnt += 1
                        '''

                        if dist_pos < safe_dist ** 2:
                            if verbose:
                                print('Collision found with ego pos {} and agent pos {}'.format(ego_p, agent_p))
                            collision_cnt += 1
                            break

                p_c = 1.0*collision_cnt/n_samples
                p_collision[i] = p_c

        if visualize:
            #plt.pause(2)
            plt.xlabel(ego_previous_action.name)
            plt.ylabel(agent_previous_action.name)
            plt.show()
            #import IPython; IPython.embed(header='risk')

        result = np.max(p_collision)

        #if result > 0.5:
        #    print('***Probability of collision:', result)

        # print()

        return result

    def normalize_theta(self, theta):
        while theta > math.radians(180):
            theta -= math.radians(180)
        while theta < math.radians(0):
            theta += math.radians(180)
        if theta > math.radians(90):
            theta = (theta * -1) + math.radians(180)

        return theta


    def values(self, state, action):                     #----------------------------------
        #approach 1
        #return sum(1 for act in action if 'stop' in act.name) + 0

        #approach 2&3  Add sqrt
        try:
            Sum = 0
            index = 0
            for agent in state.multi_state.keys():
                if agent != 'timestep':
                    if not state.multi_state[agent].turn_bool:
                        #pre_act = state.multi_state[agent].previous_action
                        if not state.multi_state[agent].turn_bool and 'stop' in action[index].name:
                            #Sum += math.sqrt(state.multi_state[agent].waiting_counter)
                            Sum += state.multi_state[agent].waiting_counter

                            #print('waiting_count', state.multi_state[agent].waiting_counter)
                        index += 1
            #print('Sum', Sum)
        except:
            IPython.embed(header='values function')
        return Sum


    def heuristic(self, state):
        return 0
    '''
    def heuristic(self, state):                     #checks direct distance from current state to goal state
        # RAO* API - estimate to the goal, or value function if receding horizon
     
        #useful to dermine if action a will take me closer to goal
        # square of euclidean distance as heuristic
        # mdeyo: found this issue! We are trying to minimize the values so the
        # heuristic should be an underestimate,which the square of distance is
        # not if each action then cost 1 or 2
        ego_x = state.get_state('Ego').state['x']
        ego_y = state.get_state('Ego').state['y']

        goal_state_keys = list(self.goal_state.keys())

        if 'x' in goal_state_keys and 'y' in goal_state_keys:
            euclidean_dist = (ego_x - self.goal_state['x']) ** 2 + (ego_y - self.goal_state['y']) ** 2
        elif 'y' in goal_state_keys:
            euclidean_dist = (ego_y - self.goal_state['y']) ** 2
        elif 'x' in goal_state_keys:
            euclidean_dist = (ego_x - self.goal_state['x']) ** 2
        else:
            assert True, 'goal state is not well defined'
            

        # print('heuristic', euclidean_dist)
        return euclidean_dist
        '''


    def execution_risk_heuristic(self, state):                     #----------------------------------
        ''' RAO* API - estimate of risk in multi-state, default 0 for admissible'''
        return 0

    def costs(self, action):                     #----------------------------------
        '''Return cost of given action, should call a cost function specific to ego vehicle model'''
        return action.cost


class VehicleModel(object):                                                                    # vehicle object, defines actions, attributes for each vehicle
    '''Individual vehicle model, this one for intersections.

    Attributes:
        name(str): Name to id each vehicle.
        current_state(dictionary): Maps variables to values for current state
            of the vehicle. Starts with initial state and is used during execution.
        attr2(: obj: `int`, optional): Description of `attr2`.

    '''

    def __init__(self, name, initial_state, model_action_list=None, isControllable=False, speed_limits=[0, 10], DetermObs=True, prerequisite=None):
        self.name = name
        self.current_state = initial_state
        self.speed_limits = speed_limits
        self.action_list = model_action_list or []
        self.current_state.name = name
        self.controllable = isControllable
        self.forward_buffer_distance = 5
        self.prerequisite = prerequisite  # Added by Rashid: for the car in front of it in intersection

    def add_action(self, action_model):              #adds an action to action list for each vehicle
        for action in self.action_list:
            if action.name == action_model.name:
                print('VehicleModel: ' + self.name +
                      ' already has action named: ' + action.name)
                return False
        action_model.agent_name = self.name
        self.action_list.append(action_model)
        return True

    def get_actions(self):
        return self.action_list

    def get_available_actions(self, multi_state, model):
        '''
        Return list of actions that meet preconditions from given multi-state
        '''
        available_actions = []
        for action in self.action_list:
            if action.preconditions(multi_state, model):
                available_actions.append(action)

        if available_actions == []:
            raise ValueError('VehicleModel for ' + self.name +
                             ' has no available actions at multi_state:' + multi_state.get_multistate_str())

        return available_actions

    def __repr__(self):
        return "VehicleModel({})".format(str(self.current_state))


class ActionModel(object):                                                         #defines each action's precondition, and exected outcome
    """Model for each vehicle action with preconditions and effects functions.

    Attributes:
        name (str): Name to id each vehicle.
        precondition_check (func): Takes instance of geordi_model, to access road model and multi-state, returns True if action can be taken for this vehicle
        effect_function (func): Takes instance of geordi_model, to access road model and multi-state, returns next state for this vehicle after this action
    """

    def __init__(self, name, action_cost=1, duration=1, p=1.0, i=0, precondition_check=lambda x: False, effect_function=lambda x: x):
        self.name = name
        self.precondition_check = precondition_check
        self.effect_function = effect_function
        self.cost = action_cost
        self.agent_name = "unassigned"
        self.length_of_action = duration
        self.probability = p
        self.index = i

    def __repr__(self):
        return 'ActionModel({})'.format(self.name)

    def set_precondition_function(self, func):
        if callable(func):
            self.precondition_check = func
        else:
            raise ValueError('not given callable precondition_check')

    def set_effect_function(self, func):
        if callable(func):
            self.effect_function = func

    def preconditions(self, state, model):
        return self.precondition_check(self.agent_name, state, model)

    def effects(self, state, model):
        return self.effect_function(self.agent_name, state, model)


####################
# Example action models
####################

def agent_forward_proximity_safe(name, this_state, length_of_action, multi_state, model):                    #-------------------------
    '''Function used for uncontrollable agent vehicles to check forward proximity. Will return False
    if the agent might get too close, defined by forward_buffer_distance for each vehicleModel.
    Does a check on current positions and a linearized projection based on current velocities
    and the duration of the action. Vehicle in front of agent with zero velocity should be treated
    different than vehicle currently moving at same speed.'''

    forward_buffer = model.get_vehicle_model(name).forward_buffer_distance
    for vehicle in multi_state.get_vehicle_state_list():
        if vehicle.name != name:
            if vehicle.state['lane_num'] == this_state['lane_num'] and vehicle.state['x'] > this_state['x']:
                # other vehicle is in front of this_vehicle
                if vehicle.state['x'] < (this_state['x'] + forward_buffer):
                    # Too close to the vehicle in front of it
                    return False
                this_projection = this_state['x'] + \
                    this_state['v'] * length_of_action
                other_projection = vehicle.state['x'] + \
                    vehicle.state['v'] * length_of_action
                if other_projection < (this_projection + forward_buffer):
                    # Too close if each vehicle stays at current velocities
                    return False
    return True


'''TODO


mdeyo:
Inefficient and wasteful to call these proximity functions for each maneuver for each vehicle.

Really need to only call them for each agent vehicle once, and then use their results
for pruning actions at a given MultiState.
'''


def agent_right_proximity_safe(name, this_state, length_of_action, multi_state, model):
    '''Function used for uncontrollable agent vehicles to check right side proximity
    before changing lanes. Will return False if the agent might get too close, defined
    by forward_buffer_distance for each vehicleModel.
    Does a check on current positions and a linearized projection based on current velocities
    and the duration of the action. Vehicle in front of agent with zero velocity should be treated
    '''
    merge_buffer = model.get_vehicle_model(name).merge_buffer_distance
    right_lane_num = model.get_lane_right(this_state['lane_name'])
    for vehicle in multi_state.get_vehicle_state_list():
        if vehicle.name != name:
            if vehicle.state['lane_num'] != right_lane_num and \
                    vehicle.state['x'] > this_state['x'] and \
                    vehicle.state['x'] < (this_state['x'] + forward_buffer):                      #typo
                # Too close to the vehicle in front of it
                return False
    return True

                                                                              #defintion of each action 
def move_forward_action(ego=False):
    action_model = ActionModel("move forward", 1)
    length_of_action = 1

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # vehicle whose action is being considered for validity
        this_state = multi_state.get_state(name).state

        # check that current road lane is clear for the next vel*deltaT units
        lane_look_ahead = this_state['x'] + \
            this_state['v'] * length_of_action

        if not model.road_model.valid_forward(lane_look_ahead, this_state['lane_num']):
            return False

        # If not our controllable vehicle - prune actions based on proximity
        # to cause behavior like slowing down when close to vehicles
        if not ego:
            if not agent_forward_proximity_safe(name, this_state, length_of_action, multi_state, model):
                return False

        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # Do stuff with the current_state to get the next state !!
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        new_state['x'] = this_state['x'] + this_state['v'] * length_of_action

        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model


def merge_left_action(ego=False):
    length_of_action = 1
    action_model = ActionModel("merge left", 2, duration=length_of_action)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # vehicle whose action is being considered for validity
        this_state = multi_state.get_state(name).state

        # check that current road lane is clear for the next vel*deltaT units
        lane_look_ahead = this_state['x'] + \
            this_state['v'] * length_of_action

        if not model.road_model.valid_forward(lane_look_ahead, this_state['lane_num']):
            return False

        if not model.road_model.left_open(this_state['x'], this_state['lane_num']):
            return False

        if not model.road_model.left_open(lane_look_ahead, this_state['lane_num']):
            return False

        # If not our controllable vehicle - prune actions based on proximity
        # to cause behavior like slowing down when close to vehicles
        if not ego:
            if not agent_forward_proximity_safe(name, this_state, length_of_action, multi_state, model):
                return False

        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        # Do stuff with the current_state to get the next state !!
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        new_state['x'] = this_state['x'] + this_state['v'] * length_of_action
        new_state['lane_num'] = this_state['lane_num'] - 2
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model


def merge_right_action(ego=False):
    length_of_action = 1
    action_model = ActionModel("merge right", 2, duration=length_of_action)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # vehicle whose action is being considered for validity
        this_state = multi_state.get_state(name).state

        # check that current road lane is clear for the next vel*deltaT units
        lane_look_ahead = this_state['x'] + \
            this_state['v'] * length_of_action

        if not model.road_model.valid_forward(lane_look_ahead, this_state['lane_num']):
            return False

        if not model.road_model.right_open(this_state['x'], this_state['lane_num']):
            return False

        if not model.road_model.right_open(lane_look_ahead, this_state['lane_num']):
            return False

        # If not our controllable vehicle - prune actions based on proximity
        # to cause behavior like slowing down when close to vehicles
        if not ego:
            if not agent_forward_proximity_safe(name, this_state, length_of_action, multi_state, model):
                return False

        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        new_state['x'] = this_state['x'] + this_state['v'] * length_of_action
        new_state['lane_num'] = this_state['lane_num'] + 2
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model


#################################
# Lane Change Actions
#################################
def ego_forward_action(ego=False):
    length_of_action = 1
    action_model = ActionModel("ego_forward", 0, duration=length_of_action)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        new_state['x'] = this_state['x'] + 30
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model

def ego_slow_forward_action(ego=False):
    length_of_action = 1
    action_model = ActionModel("ego_slow_forward", 2, duration=length_of_action)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        new_state['x'] = this_state['x'] + 20
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model

def ego_merge_right_action(ego=False):
    length_of_action = 1
    action_model = ActionModel("ego_merge_right", 1, duration=length_of_action)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # vehicle whose action is being considered for validity
        this_state = multi_state.get_state(name).state

        if this_state['lane_num'] != 1:
            return False

        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        new_state['y'] = this_state['y'] + 3
        new_state['x'] = this_state['x'] + 30
        new_state['lane_num'] = 3
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model

def ego_merge_left_action(ego=False):
    length_of_action = 1
    action_model = ActionModel("ego_merge_left", 1, duration=length_of_action)

    def preconditions(name, multi_state, model):                                  #precondition if i am at lane 3 'right lane, then i can go left
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # vehicle whose action is being considered for validity
        this_state = multi_state.get_state(name).state
        # print('((((((((((((((((((((((', this_state)

        if this_state['lane_num'] != 3:
            return False

        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")                        #effects are predefined
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        new_state['y'] = this_state['y'] - 3
        new_state['x'] = this_state['x'] + 30
        new_state['lane_num'] = 1
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model

def agent_slow_forward_action(action_probability, index, ego=False):
    length_of_action = 1
    action_model = ActionModel("agent_slow_forward"+str(index), 1, duration=length_of_action, p=action_probability, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        new_state['x'] = this_state['x'] + 20
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model

def agent_merge_right_action(action_probability, index, ego=False):
    length_of_action = 1
    action_model = ActionModel("agent_merge_right"+str(index), 1, duration=length_of_action, p=action_probability, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # vehicle whose action is being considered for validity
        this_state = multi_state.get_state(name).state

        # if this_state['lane_num'] != 1:
        #     return False

        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        new_state['y'] = this_state['y'] + 3
        new_state['x'] = this_state['x'] + 20
        # new_state['lane_num'] = 3
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model

def agent_merge_left_action(action_probability, index, ego=False):
    length_of_action = 1
    action_model = ActionModel("agent_merge_left"+str(index), 1, duration=length_of_action, p=action_probability, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # vehicle whose action is being considered for validity
        this_state = multi_state.get_state(name).state

        # if this_state['lane_num'] != 3:
        #     return False

        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        new_state['y'] = this_state['y'] - 3
        new_state['x'] = this_state['x'] + 20
        # new_state['lane_num'] = 1
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model


#################################
# Left Turn Actions
#################################


def agent_forward_action(action_probability, index, ego=False):
    length_of_action = 1
    action_model = ActionModel("agent_forward"+str(index), 1, duration=length_of_action, p=action_probability, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        new_state['y'] = this_state['y'] - 60
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model

def agent_slow_down_action(action_probability, index, ego=False):
    length_of_action = 1
    action_model = ActionModel("agent_slow_down"+str(index), 1, duration=length_of_action, p=action_probability, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        new_state['y'] = this_state['y'] - 20
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model

def turn_left_action(ego=False):
    length_of_action = 1
    action_model = ActionModel("ego_turn_left", 0, duration=length_of_action)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        new_state['x'] = this_state['x'] + 12
        new_state['y'] = this_state['y'] + 20
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model


def stop_action(ego=False):                                               #
    length_of_action = 1
    action_model = ActionModel("ego_stop", 1, duration=length_of_action)

    # no preconditions for stop action
    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        return True

    # no changes (as a null action)
    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model


#######################################################################################################################
# Intersection Actions
#######################################################################################################################

#each side will have a single definition for left, forward, and right
#total of 12 actions +stop action
#left side and right side doesn't matter, as they share the forward action and it's similar
#cost: stop is one, rest of actions is zero



def intersect_stop_action(ego=False):
    length_of_action = 1
    action_model = ActionModel("stop", 1, duration=length_of_action)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        new_state['x'] = this_state['x']
        new_state['y'] = this_state['y']
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model

# actions are slow and fast

#Slow

#Side A

def side_A_left_left_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_A_left_left", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 16.5
        #new_state['y'] = this_state['y'] - 17
        new_state['x'] = 31
        new_state['y'] = 74.5
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_A_left_forward_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_A_left_forward", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 29
        #new_state['y'] = this_state['y'] + 0
        new_state['x'] = 43.5
        new_state['y'] = 91.5
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_A_right_forward_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_A_right_forward", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 29
        #new_state['y'] = this_state['y'] + 0
        new_state['x'] = 43.5
        new_state['y'] = 95
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model




def side_A_right_right_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_A_right_right", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 9.5
        #new_state['y'] = this_state['y'] + 13
        new_state['x'] = 24
        new_state['y'] = 108
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model





#Side B

def side_B_left_left_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_B_left_left", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] - 17
        #new_state['y'] = this_state['y'] + 15.5
        new_state['x'] = 28
        new_state['y'] = 104
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_B_left_forward_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_B_left_forward", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] - 29.5
        #new_state['y'] = this_state['y'] + 0
        new_state['x'] = 15.5
        new_state['y'] = 88.5
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_B_right_forward_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_B_right_forward", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] - 29.5
        #new_state['y'] = this_state['y'] + 0
        new_state['x'] = 15.5
        new_state['y'] = 85
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model




def side_B_right_right_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_B_right_right", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] - 9.5
        #new_state['y'] = this_state['y'] - 10.5
        new_state['x'] = 35.5
        new_state['y'] = 74.5
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model









#Side C

def side_C_left_left_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_C_left_left", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 16
        #new_state['y'] = this_state['y'] + 19.5
        new_state['x'] = 44
        new_state['y'] = 92
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_C_left_forward_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_C_left_forward", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 0
        #new_state['y'] = this_state['y'] + 32.5
        new_state['x'] = 28
        new_state['y'] = 105
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_C_right_forward_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_C_right_forward", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 0
        #new_state['y'] = this_state['y'] + 32.5
        new_state['x'] = 24.5
        new_state['y'] = 105
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_C_right_right_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_C_right_right", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] - 9
        #new_state['y'] = this_state['y'] + 11.5
        new_state['x'] = 15.5
        new_state['y'] = 84
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model








#Side D

def side_D_left_left_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_D_left_left", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] - 16
        #new_state['y'] = this_state['y'] - 17
        new_state['x'] =15.5
        new_state['y'] =88
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_D_left_forward_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_D_left_forward", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 0
        #new_state['y'] = this_state['y'] - 31
        new_state['x'] = 31.5
        new_state['y'] = 74
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_D_right_forward_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_D_right_forward", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 0
        #new_state['y'] = this_state['y'] - 31
        new_state['x'] = 35
        new_state['y'] = 74
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model




def side_D_right_right_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_D_right_right", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 9
        #new_state['y'] = this_state['y'] - 9
        new_state['x'] = 44
        new_state['y'] = 96
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



#Fast

#Side A

def side_A_left_left_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_A_left_left_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 16.5
        #new_state['y'] = this_state['y'] - 17
        new_state['x'] = 31
        new_state['y'] = 74.5
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_A_left_forward_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_A_left_forward_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 29
        #new_state['y'] = this_state['y'] + 0
        new_state['x'] = 43.5
        new_state['y'] = 91.5
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_A_right_forward_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_A_right_forward_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 29
        #new_state['y'] = this_state['y'] + 0
        new_state['x'] = 43.5
        new_state['y'] = 95
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model




def side_A_right_right_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_A_right_right_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 9.5
        #new_state['y'] = this_state['y'] + 13
        new_state['x'] = 24
        new_state['y'] = 108
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model





#Side B

def side_B_left_left_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_B_left_left_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] - 17
        #new_state['y'] = this_state['y'] + 15.5
        new_state['x'] = 28
        new_state['y'] = 104
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_B_left_forward_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_B_left_forward_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] - 29.5
        #new_state['y'] = this_state['y'] + 0
        new_state['x'] = 15.5
        new_state['y'] = 88.5
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_B_right_forward_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_B_right_forward_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] - 29.5
        #new_state['y'] = this_state['y'] + 0
        new_state['x'] = 15.5
        new_state['y'] = 85
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model




def side_B_right_right_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_B_right_right_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] - 9.5
        #new_state['y'] = this_state['y'] - 10.5
        new_state['x'] = 35.5
        new_state['y'] = 74.5
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model









#Side C

def side_C_left_left_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_C_left_left_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 16
        #new_state['y'] = this_state['y'] + 19.5
        new_state['x'] = 44
        new_state['y'] = 92
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_C_left_forward_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_C_left_forward_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 0
        #new_state['y'] = this_state['y'] + 32.5
        new_state['x'] = 28
        new_state['y'] = 105
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_C_right_forward_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_C_right_forward_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 0
        #new_state['y'] = this_state['y'] + 32.5
        new_state['x'] = 24.5
        new_state['y'] = 105
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_C_right_right_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_C_right_right_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] - 9
        #new_state['y'] = this_state['y'] + 11.5
        new_state['x'] = 15.5
        new_state['y'] = 84
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model








#Side D

def side_D_left_left_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_D_left_left_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] - 16
        #new_state['y'] = this_state['y'] - 17
        new_state['x'] =15.5
        new_state['y'] =88
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_D_left_forward_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_D_left_forward_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 0
        #new_state['y'] = this_state['y'] - 31
        new_state['x'] = 31.5
        new_state['y'] = 74
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model



def side_D_right_forward_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_D_right_forward_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 0
        #new_state['y'] = this_state['y'] - 31
        new_state['x'] = 35
        new_state['y'] = 74
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model




def side_D_right_right_fast_action(ego=False, index=0):
    length_of_action = 1
    action_model = ActionModel("side_D_right_right_fast", 0, duration=length_of_action, i=index)

    def preconditions(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")

        # no preconditions
        return True

    def effects(name, multi_state, model):
        if not isinstance(model, GeordiModel):
            raise TypeError("input must be a GeordiModel")
        this_state = multi_state.get_state(name).state
        new_state = this_state.copy()
        #new_state['x'] = this_state['x'] + 9
        #new_state['y'] = this_state['y'] - 9
        new_state['x'] = 44
        new_state['y'] = 96
        return VehicleState(state=new_state, name=name, previous_action=action_model.name)

    action_model.set_precondition_function(preconditions)
    action_model.set_effect_function(effects)
    return action_model





