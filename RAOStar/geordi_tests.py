#!/usr/bin/env python

# author: Matt Deyo
# email: mdeyo@mit.edu
# simple vehicle models based on MERS Toyota work for rao star

from utils import import_models
import_models()
from vehicle_model import *
from geordi_road_model import *
from raostar import RAOStar

road_model = highway_2_lanes_offramp_ex()
print(road_model)
# plot_road_model(road_model)

ego_vehicle = VehicleModel('Ego', VehicleState(
    state={'lane_num': 1, 'x': 0, 'y': 0, 'v': 5, 'theta': 0}), isControllable=True)

agent1_vehicle = VehicleModel('Agent1', VehicleState(
    state={'lane_num': 3, 'x': 0, 'y': 0, 'v': 5, 'theta': 0}))

agent2_vehicle = VehicleModel('Agent2', VehicleState(
    state={'lane_num': 3, 'x': 5, 'y': 0, 'v': 5, 'theta': 0}))

ego_vehicle.add_action(move_forward_action(ego=True))
ego_vehicle.add_action(merge_left_action(ego=True))
ego_vehicle.add_action(merge_right_action(ego=True))

agent1_vehicle.add_action(move_forward_action())
agent2_vehicle.add_action(move_forward_action())
agent2_vehicle.add_action(merge_left_action())
agent2_vehicle.add_action(merge_right_action())

# geordi_model = GeordiModel()
geordi_model = GeordiModel(
    [ego_vehicle, agent1_vehicle, agent2_vehicle], road_model)
print(geordi_model.road_model)
actions = geordi_model.get_available_actions(geordi_model.current_state)
print(actions)
print(agent1_vehicle.action_list[0].precondition_check(agent1_vehicle.name,
                                                       geordi_model.current_state, geordi_model))

new_states = geordi_model.state_transitions(
    geordi_model.current_state, actions[0])

print('new_states', new_states)

algo = RAOStar(geordi_model, cc=0.1, debugging=True, cc_type='o')

b_init = {geordi_model.current_state: 1.0}
P, G = algo.search(b_init)
print(P)

# permutations = calculate_permutations(
#     geordi_model, geordi_model.current_state, agent2_vehicle.action_list, 5)

# print(len(permutations))
# print(permutations)

# for p in permutations:
# print(p['actions'])

# geordi_model.add_vehicle_model(ego_vehicle)
# geordi_model.add_vehicle_model(agent_vehicle1)
