#!/usr/bin/env python

# author: Cyrus Huang
# email: huangxin@mit.edu
# intention-aware vehicle models with probabilistic flow tubes

import numpy as np
import copy
from pprint import pformat
import pickle
import matplotlib.pyplot as plt

import sys
import os
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


class VehicleState(object):
    '''Object to represent a single vehicle state in the Geordi Model. Includes
    position, velocity, and additional attributes to describe a given state'''

    def __init__(self, name="Unnamed", state=None, previous_action=None):
        self.name = name
        self.state = state
        self.previous_action = previous_action

    def __repr__(self):
        info = {'name': self.name, 'state': self.state, 'pre-action':self.previous_action}
        # return pformat(info)
        return 'VehicleState(name: {}, state: {}, previous_action: {})'.format(self.name,str(self.state),self.previous_action)

class MultiState(object):
    '''Object to store composite state of the world in Geordi Model. Has a
    dictionary that contains the 'vehicle-name': 'vehicle-state' for all the
    vehicles in a given multi-state. Also has utility functions like
    get_state(vehicle-name) and print_multistate()'''

    def __init__(self, timestep=-1):
        self.timestep = timestep
        self.multi_state = {}
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


class GeordiModel(object):
    '''First Python version of Geordi vehicles model, this one for intersections.

    Attributes:
        name (str): Name to id each vehicle.
        current_state (dictionary): Maps variables to values for current state
        of the vehicle. Starts with initial state and is used during execution.
        attr2 (:obj:`int`, optional): Description of `attr2`.
    '''

    def __init__(self, vehicle_models=[], road=None, goal_state=(0,0)):
        print('Made GeordiModel! 2.0')
        self.optimization = 'minimize'  # want to minimize the steps to goal

        self.vehicle_models = {}
        self.road_model = road
        self.controllable_vehicle = None
        # Start with MultiState at timestep=0
        self.current_state = MultiState(0)
        self.goal_state = goal_state
        # For all vehicle_models: add to MultiState and dict of vehicles
        for i in vehicle_models:
            self.current_state.add_vehicle_state(i.current_state)
            self.vehicle_models[i.name] = i
            print(i)
            if i.controllable:
                self.controllable_vehicle = i
        print("GeordiModel MultiState: ",
              self.current_state.get_multistate_str())

    def add_vehicle_model(self, vehicle_model):
        new_name = vehicle_model.name
        if new_name not in self.vehicle_models:
            # If not saved in model, add to vehicle list and MultiState
            self.vehicle_models[new_name] = vehicle_model
            self.current_state.add_vehicle_state(i.current_state)
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

    def get_vehicle_model(self, name):
        if name in self.vehicle_models:
            return self.vehicle_models[name]

    def get_vehicle_state(self, name):
        return self.get_vehicle_mode(name).name

    def get_available_actions(self, multi_state):
        '''
        Return set of controllable actions available from given multi-state
        This list of actions is just pruned by the road model itself, not nearby
        vehicles. We are not enforcing safe driving through this filter.
        '''
        our_vehicle = self.controllable_vehicle
        available_actions = []
        for action in our_vehicle.get_actions():
            # print('assessing action: ' + str(action))
            # print(action.precondition_check)
            if action.preconditions(multi_state, self):
                available_actions.append(action)

        return available_actions
    '''
    ##  API required by RAO*  ##
        self.A = model.actions  # takes state, and given all possible actions
        self.T = model.state_transitions  # T(S,A) = S'
        self.O = model.observations  # takes state, and gives all possible observation with probability
        self.V = model.values  # reward or cost of S,A (takes husristics into considration)
        self.h = model.heuristic  # not used
        self.r = model.state_risk  # risk of given state
        self.e = model.execution_risk_heuristic  # not used
        self.term = model.is_terminal  # determines if state is terminal
    '''

    def observations(self, state):
        return [(state, 1.0)]

    def actions(self, state):
        '''RAO* API - return list of available_actions in given multi-state state'''
        return self.get_available_actions(state)

    def state_transitions(self, state, action):
        '''RAO* API - return distribution of multi-states that result from
            taking given action in given multi-state'''

        # print('state_transitions', state, action)

        new_ego_vehicle_state = action.effects(state, self)
        timestep = state.timestep
        duration = action.length_of_action

        # list of lists that have state and probability,
        # for example: [[state_2, prob],[state_3, prob]]
        # new_state_distribution = []

        # print(len(self.vehicle_models))

        # Edge case: assume it's just our controllable vehicle??
        if len(self.vehicle_models) == 1:
            new_multi_state = MultiState(timestep + duration)
            new_multi_state.add_vehicle_state(new_ego_vehicle_state)
            return [[new_multi_state, 1.0]]

        # Regular permutation calculations
        state_distribution = [{'prob': 1, 'states': [new_ego_vehicle_state]}]

        for vehicle_name, vehicle_model in self.vehicle_models.items():
            # print("*****Name", vehicle_name)
            # print('each vehicle:', vehicle_name)
            # If selected vehicle is the controllable_vehicle
            if vehicle_model == self.controllable_vehicle:
                # Don't further permutate over our actions
                continue

            new_state_distribution = []

            available_actions = vehicle_model.get_available_actions(state, self)
            # assume next action is always going forward
            if state.timestep >= 1:
                for action in available_actions:
                    if action.name == 'forward':
                        available_actions = [action]
                        break

            for action in available_actions:
                # print("***Action",action, action.name, action.probability)
                # print(state_distribution)
                new_state = action.effects(state, self)
                for combo in state_distribution:
                    new_combo = copy.deepcopy(combo)
                    new_combo['states'].append(new_state)
                    # print("------------------------------")
                    # print(new_state)
                    new_combo['prob'] = new_combo['prob'] * action.probability
                    new_state_distribution.append(new_combo)
                # print(new_state)
                # print(new_state.previous_action)

            state_distribution = new_state_distribution
        # print(state_distribution)
        # print("Number of states at next level", len(state_distribution))

        sum_probs = sum(c['prob'] for c in state_distribution)

        new_states = []
        for outcome in state_distribution:
            new_multi_state = MultiState(timestep + duration)
            for agent in outcome['states']:
                new_multi_state.add_vehicle_state(agent)
            new_states.append([new_multi_state, outcome['prob'] / sum_probs])

        return new_states

    def goal_function(self, state):
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

    def is_terminal(self, state):
        '''RAO* API - return True if this state is terminal, either planning
        horizon edge, goal state, or terminal failure'''
        # if self.iterative_deepening:
        # if state.depth == self.planning_horizon:
        # return True
        if self.goal_function(state):
            return True

    def state_risk(self, state):
        '''RAO* API - return 1.0 if multi-state breaks constraint'''
        # Compute risk as a probability of colliding with other vehicles
        # using probabilistic flow tubes representing vehicle actions

        # get ego vehicle state
        ego_state = (state.get_state('Ego').state['x'], 
                     state.get_state('Ego').state['y'],
                     state.get_state('Ego').state['yaw']) 
        ego_previous_action = state.get_state('Ego').previous_action
        # print(ego_state, ego_previous_action)

        # compute collision probability with each agent vehicle
        # assume all agent vehicles have names starting with 'Agent'
        multi_state_keys = state.multi_state.keys()
        agent_names = list(filter(lambda x:x.startswith(('Agent')), multi_state_keys))
        p_collision = np.array([0.0]*len(agent_names))

        for i, agent in enumerate(agent_names):
            agent_state = (state.get_state(agent).state['x'], 
                           state.get_state(agent).state['y'],
                           state.get_state(agent).state['yaw'])
            agent_previous_action = state.get_state(agent).previous_action

            if ego_previous_action == 'ego_stop' or not ego_previous_action:
                p_collision[i] = 0.0
                continue

            # get clock for previous action
            actions = self.get_vehicle_model(agent).action_list
            agent_index = 0
            for action in actions:
                if action.name == agent_previous_action:
                    # agent_index = action.index
                    agent_index = action.index

            collision_result = self.compute_pairwise_action_risk(ego_state,
                                                                 # ego_index, # TODO: change
                                                                 0,
                                                                 ego_previous_action,
                                                                 agent_state,
                                                                 agent_index,
                                                                 agent_previous_action)
            p_collision[i] = collision_result
            # print('check point', collision_result, p_collision)

        # compute collision probability with any agent vehicle
        risk = 1.0 - np.prod((1.0 - p_collision))
        # print('Risks: {}, final risk: {}'.format(p_collision, risk))
        return risk

    def load_pft(self, action_name):
        action_name = ''.join([i for i in action_name if not i.isdigit()])
        pft_path = os.path.join(PFT_DATA_DIR, '%s_pft_short.pkl' % (action_name))
        with open(pft_path, 'rb') as f_snap:
            pft = pickle.load(f_snap)
            # print(action_name, pft.pos_mu)
        return pft

    def compute_pairwise_action_risk(self, ego_state, ego_index, ego_previous_action, agent_state, agent_index, agent_previous_action):
        verbose = False
        visualize = False
        safe_dist = 2
        n_samples = 100
        # get ego and agent pfts
        ego_pft = self.load_pft(ego_previous_action)
        agent_pft = self.load_pft(agent_previous_action)

        # print('Ego info:', ego_state, ego_previous_action, ego_pft.l)
        # print('Agent info:', agent_state,agent_previous_action, agent_pft.l)

        # assume pft's have equal lengths
        assert ego_pft.l == agent_pft.l, '{} - {}, {} - {}'.format(ego_previous_action, ego_pft.l, agent_previous_action, agent_pft.l)
        ego_pft_mu = ego_pft.pos_mu
        ego_pft_sigma = ego_pft.pos_sigma
        agent_pft_mu = agent_pft.pos_mu
        agent_pft_sigma = agent_pft.pos_sigma

        # a quick hack
        # TODO: fix this
        if agent_index >= len(agent_pft_mu):
            agent_index = len(agent_pft_mu) - 1

        # compute starting position of each vehicle given their actions
        ego_start = (ego_state[0] - (ego_pft_mu[-1][0] - ego_pft_mu[ego_index][0]),
                     ego_state[1] - (ego_pft_mu[-1][1] - ego_pft_mu[ego_index][1]))
        agent_start = (agent_state[0] - (agent_pft_mu[-1][0] - agent_pft_mu[agent_index][0]),
                       agent_state[1] - (agent_pft_mu[-1][1] - agent_pft_mu[agent_index][1]))


        # print('ego:', ego_start, ego_state)
        # print('agent:', agent_start, agent_state, agent_index)

        if visualize:
            plt.plot(ego_state[0], ego_state[1], 'rx')
            plt.plot(agent_state[0], agent_state[1], 'bx')
            plt.plot(ego_start[0], ego_start[1], 'ro')
            plt.plot(agent_start[0], agent_start[1], 'bo')

        p_collision = np.array([0.0]*ego_pft.l)
        for i in range(ego_pft.l):
            # compute shifted positions
            if i + ego_index < ego_pft.l:
                ego_mu = (ego_pft_mu[i + ego_index][0] + ego_start[0],
                          ego_pft_mu[i + ego_index][1] + ego_start[1])
                ego_sigma = ego_pft_sigma[i + ego_index]
            else:
                continue

            if i + agent_index < agent_pft.l:
                agent_mu = (agent_pft_mu[i + agent_index][0] + agent_start[0],
                            agent_pft_mu[i + agent_index][1] + agent_start[1])
                agent_sigma = agent_pft_sigma[i + agent_index]
            else:
                continue

            if visualize:
                plt.plot(ego_mu[0],ego_mu[1],'r.')
                plt.plot(agent_mu[0],agent_mu[1],'b.')

            # compute distance between pairwise locations
            dist = (ego_mu[0] - agent_mu[0]) ** 2 + (ego_mu[1] - agent_mu[1]) ** 2

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
                        if dist_pos < safe_dist ** 2:
                            if verbose:
                                print('Collision found with ego pos {} and agent pos {}'.format(ego_p, agent_p))

                            collision_cnt += 1
                            break

                p_c = 1.0*collision_cnt/n_samples
                p_collision[i] = p_c

        if visualize:
            plt.show()

        result = np.max(p_collision)    
        # print('***Probability of collision:', result)
        # print()

        return result

    def values(self, state, action):
        '''RAO* API - return value of a state (heuristic + cost)'''
        # return value (heuristic + cost)
        return self.costs(action)
        return self.costs(action) + self.heuristic(state)

    def heuristic(self, state):
        '''RAO* API - estimate to the goal, or value function if receding horizon'''
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

    def execution_risk_heuristic(self, state):
        ''' RAO* API - estimate of risk in multi-state, default 0 for admissible'''
        return 0

    def costs(self, action):
        '''Return cost of given action, should call a cost function specific to ego vehicle model'''
        return action.cost


class VehicleModel(object):
    '''Individual vehicle model, this one for intersections.

    Attributes:
        name(str): Name to id each vehicle.
        current_state(dictionary): Maps variables to values for current state
            of the vehicle. Starts with initial state and is used during execution.
        attr2(: obj: `int`, optional): Description of `attr2`.

    '''

    def __init__(self, name, initial_state, model_action_list=None, isControllable=False, speed_limits=[0, 10], DetermObs=True):
        self.name = name
        self.current_state = initial_state
        self.speed_limits = speed_limits
        self.action_list = model_action_list or []
        self.current_state.name = name
        self.controllable = isControllable
        self.forward_buffer_distance = 5

    def add_action(self, action_model):
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


class ActionModel(object):
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

def agent_forward_proximity_safe(name, this_state, length_of_action, multi_state, model):
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
                    vehicle.state['x'] < (this_state['x'] + forward_buffer):
                # Too close to the vehicle in front of it
                return False
    return True


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

    def preconditions(name, multi_state, model):
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
            raise TypeError("input must be a GeordiModel")
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


def stop_action(ego=False):
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