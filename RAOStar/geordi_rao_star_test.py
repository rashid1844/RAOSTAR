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
import numpy as np
import random
import matplotlib.pyplot as plt

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


# get pft data
test = 'agent'
maneuvers = ['forward', 'slow_down']
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PFT_DIR = os.path.join(WORKSPACE_DIR,'PFT')
sys.path.insert(0, PFT_DIR)
PROCESSED_DATA_DIR = os.path.join(PFT_DIR,'pft_data')

pfts = []
for m in maneuvers:
    processed_pft_data_path = os.path.join(PROCESSED_DATA_DIR, '%s_%s_pft_short.pkl' % ('agent', m))
    with open(processed_pft_data_path, 'rb') as f_snap:
        pft = pickle.load(f_snap)
        pfts.append(pft)

# or read trajectories from human demonstrations
trajectories = {}
for maneuver in maneuvers:
    processed_data_path = os.path.join(PROCESSED_DATA_DIR, '%s_%s_raw.pkl' % (test, maneuver))
    print(processed_data_path)
    with open(processed_data_path, 'rb') as f_snap:
        data = pickle.load(f_snap)

    trajectories[maneuver] = data


delta = [0.99, 0.5, 0.1,0.001,0.0001,0.00001]
# delta = [0.1]

total_crashes = []
total_time = []
for d in delta:
    crashes = []
    times = []
    test_maneuvers = []
    for i in range(100):

        # get a random testing trajectory
        p = random.uniform(0,1)
        m_idx = np.random.choice([0,1], p=[p, 1-p])
        # m_idx = 0
        maneuver = maneuvers[m_idx]
        test_maneuvers.append(maneuver)
        print(maneuver)

        trajs = trajectories[maneuver]

        if m_idx == 0:
            traj = np.array(trajs[4]).T
            # traj = np.array(random.choice([trajs[2],trajs[4]])).T
        else:
            traj = np.array(random.choice(trajs)).T
        traj = traj[::10,:]

        # plt.plot(traj[:,1],traj[:,2],'rx')
        # plt.show()

        p_m = np.array([0.5, 0.5])
        crash = 0

        for j in range(10,traj.shape[0]):
            obs = traj[j-10:j,1:3]
            print(obs.shape)

            p_m_new = np.array([0.5,0.5])        
            for k, m in enumerate(maneuvers):
                pft = pfts[k]
                best_index, likelihood, best_pos = pft.calculateMostLikelyPointPDF(obs)
                p_m_new[k] = likelihood

            p_m[0] = p_m[0] * p_m_new[0]
            p_m[1] = p_m[1] * p_m_new[1]
            p_m = p_m_new

            p_m = p_m/np.sum(p_m)
            print(p_m)


            # assume an open road
            road_model = intersection_left_turn_ex()

            # add an ego vehicle that can choose from stop (wait) and turn
            ego_vehicle = VehicleModel('Ego', VehicleState(
                state={'x': 88, 'y': 180, 'yaw': 90}), isControllable=True)
            ego_vehicle.add_action(stop_action(ego=True))
            ego_vehicle.add_action(turn_left_action(ego=True))

            # add an agent vehicle that can go forward or slow down
            agent1_vehicle = VehicleModel('Agent1', VehicleState(
                state={'x': 92, 'y': 240, 'yaw': 270}))
            # probability distribution of forward action
            # p = random.uniform(0,1)
            agent1_vehicle.add_action(agent_forward_action(p_m[0]))
            agent1_vehicle.add_action(agent_slow_down_action(p_m[1]))

            # geordi_model = GeordiModel()
            geordi_model = GeordiModel(
                [ego_vehicle, agent1_vehicle], road_model, goal_state=(100,200))
            # print(geordi_model.road_model)
            actions = geordi_model.get_available_actions(geordi_model.current_state)
            # print(actions)
            # print(agent1_vehicle.action_list[0].precondition_check(agent1_vehicle.name,
            #                                                        geordi_model.current_state, geordi_model))

            new_states = geordi_model.state_transitions(
                geordi_model.current_state, actions[0])

            # print('new_states', new_states)

            algo = RAOStar(geordi_model, cc=d, debugging=False, cc_type='o', fixed_horizon = 3)

            b_init = {geordi_model.current_state: 1.0}
            P, G = algo.search(b_init)

            print(P)
            if len(P.keys()) == 1 and m_idx == 0:
                print("Crashed")
                crash = 1
                j = 0
                break
            
            if len(P.keys()) == 1:
                print("Turn safely")
                break

        times.append((j+1)/30.0*4.8)
        crashes.append(crash)

    crashes = np.array(crashes)
    times = np.array(times)
    total_crashes.append(np.sum(crashes)*1.0/len(crashes))
    total_time.append(np.mean(times))

print(total_crashes)
print(total_time)
