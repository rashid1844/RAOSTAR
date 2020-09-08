#!/usr/bin/env python

# author: Cyrus Huang
# email: huangxin@mit.edu
# simple vehicle left turn scenario used for ICAPS19-intention 

from utils import import_models
import_models()
from intention_vehicle_model import *
from geordi_road_model import *
from raostar import RAOStar
from pprint import pprint
import ast

def prettyprint(policy):
	for keys, values in policy.items():
		state, probability, depth = keys
		best_action = values

		node_info = {}
		node_info['state'] = state
		node_info['probability'] = probability
		node_info['depth'] = depth
		node_info['the_best_action'] = best_action

		# print(ast.literal_eval(state))

		pprint(node_info)

# assume an open road
road_model = highway_2_lanes()

# add an ego vehicle that can choose from stop (wait) and turn
ego_vehicle = VehicleModel('Ego', VehicleState(
    state={'x': 40.80, 'y': 7.76, 'yaw': 0, 'lane_num':3}), isControllable=True)
ego_vehicle.add_action(ego_forward_action(ego=True))
ego_vehicle.add_action(ego_merge_left_action(ego=True))
ego_vehicle.add_action(ego_merge_right_action(ego=True))

# add an agent vehicle that can go forward or slow down
agent1_vehicle = VehicleModel('Agent1', VehicleState(
    state={'x': 47.51, 'y': 7.76, 'yaw': 0, 'lane_num':3}))
# agent1_vehicle.add_action(agent_merge_left_action(0.1))
# agent1_vehicle.add_action(agent_merge_right_action(0.1))
agent1_vehicle.add_action(agent_slow_forward_action(0.99,index=0))
# agent1_vehicle.add_action(agent_merge_left_action(0.01,index=0))

# agent2_vehicle = VehicleModel('Agent2', VehicleState(
#     state={'x': -74.7, 'y': -60, 'yaw': 0, 'lane_num':1}))
# agent2_vehicle.add_action(agent_merge_left_action(0.1))
# agent2_vehicle.add_action(agent_merge_right_action(0.1))
# agent2_vehicle.add_action(agent_slow_forward_action(1.0))

# agent3_vehicle = VehicleModel('Agent3', VehicleState(
#     state={'x': -77.7, 'y': 0, 'yaw': 0, 'lane_num':3}))
# agent3_vehicle.add_action(agent_merge_left_action(1.0))
# agent3_vehicle.add_action(agent_merge_right_action(1.0))
# agent3_vehicle.add_action(agent_slow_forward_action(1.0))

# agent4_vehicle = VehicleModel('Agent4', VehicleState(
#     state={'x': -74.7, 'y': -20, 'yaw': 0, 'lane_num':1}))
# agent4_vehicle.add_action(agent_merge_left_action(0.1))
# agent4_vehicle.add_action(agent_merge_right_action(0.1))
# agent4_vehicle.add_action(agent_slow_forward_action(1.0))

# agent5_vehicle = VehicleModel('Agent5', VehicleState(
#     state={'x': -74.7, 'y': -20, 'yaw': 0, 'lane_num':1}))
# agent5_vehicle.add_action(agent_merge_left_action(0.1))
# agent5_vehicle.add_action(agent_merge_right_action(0.1))
# agent5_vehicle.add_action(agent_slow_forward_action(1.0))

# geordi_model = GeordiModel()
geordi_model = GeordiModel(
    [ego_vehicle, agent1_vehicle,], road_model, goal_state={'x':200})
print(geordi_model.road_model)
actions = geordi_model.get_available_actions(geordi_model.current_state)
print(actions)
print(agent1_vehicle.action_list[0].precondition_check(agent1_vehicle.name,
                                                       geordi_model.current_state, geordi_model))

new_states = geordi_model.state_transitions(
    geordi_model.current_state, actions[0])

print('new_states', new_states)

algo = RAOStar(geordi_model, cc=0.01, debugging=False, cc_type='o', fixed_horizon = 1)

b_init = {geordi_model.current_state: 1.0}
P, G = algo.search(b_init, iter_limit=5)

for keys, values in P.items():
    state, probability, depth = keys
    best_action = values

    node_info = {}
    node_info['state'] = state
    node_info['probability'] = probability
    node_info['depth'] = depth
    node_info['the_best_action'] = best_action

    # if depth == 0:
    #     if best_action == 'ActionModel(ego_forward)':
    #         result = 'forward'
    #     elif best_action == 'ActionModel(ego_merge_right)':
    #         result = 'merge_right'
    #     elif best_action == 'ActionModel(ego_merge_left)':
    #         result = 'merge_left'
    #     else:
    #         result = 'stop'

    pprint(node_info)
# print(P)
# prettyprint(P)

# permutations = calculate_permutations(
#     geordi_model, geordi_model.current_state, agent2_vehicle.action_list, 5)

# print(len(permutations))
# print(permutations)

# for p in permutations:
# print(p['actions'])

# geordi_model.add_vehicle_model(ego_vehicle)
# geordi_model.add_vehicle_model(agent_vehicle1)
