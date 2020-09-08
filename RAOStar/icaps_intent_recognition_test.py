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

# sample 100 trajectories from each pft
trajectories = {}
for i, pft in enumerate(pfts):
    pft_trajectories = []

    for _ in range(10):
        trajectory = []
        for idx in range(25):
            mu = pft.pos_mu[idx]
            sigma = pft.pos_sigma[idx]
            point = np.random.multivariate_normal(mu, sigma, 1)[0]
            trajectory.append(point)

        pft_trajectories.append(np.array(trajectory))

    trajectories[maneuvers[i]] = pft_trajectories


# # or read trajectories from human demonstrations
# trajectories = {}
# for maneuver in maneuvers:
#     processed_data_path = os.path.join(PROCESSED_DATA_DIR, '%s_%s_raw.pkl' % (test, maneuver))
#     print(processed_data_path)
#     with open(processed_data_path, 'rb') as f_snap:
#         data = pickle.load(f_snap)

#     trajectories[maneuver] = data

# print(trajectories)
crashes = []
times = []
test_maneuvers = []
hits = []
n_tests = 100
for i in range(n_tests):

    # get a random testing trajectory
    p = random.uniform(0,1)
    m_idx = np.random.choice([0,1], p=[p, 1-p])
    # m_idx = 1
    maneuver = maneuvers[m_idx]
    test_maneuvers.append(maneuver)
    # print(i, maneuver)

    trajs = trajectories[maneuver]


    traj = np.array(random.choice(trajs))
    # traj = (traj.T)[::10,1:3]

    # plt.plot(traj[:,0],traj[:,1],'rx')
    # plt.show()

    p_m = np.array([0.5, 0.5])
    crash = 0

    for j in range(10,traj.shape[0]):
        obs = traj[j-10:j,:]
        # print(obs.shape)
        
        p_m_new = np.array([0.5, 0.5])
        best_positions = [0,0]
        for k, m in enumerate(maneuvers):
            pft = pfts[k]
            # pft.pos_sigma = pft.pos_sigma*4
            best_index, likelihood, best_pos = pft.calculateMostLikelyPointPDF(obs)
            best_positions[k] = best_pos
            p_m_new[k] = likelihood

        p_m[0] = p_m[0] * p_m_new[0]
        p_m[1] = p_m[1] * p_m_new[1]
        # p_m = p_m_new
        p_m = p_m/np.sum(p_m)
        # print(p_m)

        # compute the number of correctly classified maneuvers
        if p_m[0] >= p_m[1] and m_idx == 0:
            hits.append(1)
        elif p_m[0] <= p_m[1] and m_idx == 1:
            hits.append(1)
        else:
            hits.append(0)

        # compute the average displacement error
        if p_m[0] >= p_m[1]:
            best_pos = best_positions[0]
            current_pos = traj[j-10,:]

            print(best_pos)
            print(current_pos)
            print(1/0)

 
hits = np.array(hits)
print(hits)
print(np.sum(hits)*1.0/len(hits))

    #     # assume an open road
    #     road_model = intersection_left_turn_ex()

    #     # add an ego vehicle that can choose from stop (wait) and turn
    #     ego_vehicle = VehicleModel('Ego', VehicleState(
    #         state={'x': 88, 'y': 180, 'yaw': 90}), isControllable=True)
    #     ego_vehicle.add_action(stop_action(ego=True))
    #     ego_vehicle.add_action(turn_left_action(ego=True))

    #     # add an agent vehicle that can go forward or slow down
    #     agent1_vehicle = VehicleModel('Agent1', VehicleState(
    #         state={'x': 92, 'y': 240, 'yaw': 270}))
    #     # probability distribution of forward action
    #     # p = random.uniform(0,1)
    #     agent1_vehicle.add_action(agent_forward_action(p_m[0]))
    #     agent1_vehicle.add_action(agent_slow_down_action(p_m[1]))

    #     # geordi_model = GeordiModel()
    #     geordi_model = GeordiModel(
    #         [ego_vehicle, agent1_vehicle], road_model, goal_state=(100,200))
    #     # print(geordi_model.road_model)
    #     actions = geordi_model.get_available_actions(geordi_model.current_state)
    #     # print(actions)
    #     # print(agent1_vehicle.action_list[0].precondition_check(agent1_vehicle.name,
    #     #                                                        geordi_model.current_state, geordi_model))

    #     new_states = geordi_model.state_transitions(
    #         geordi_model.current_state, actions[0])

    #     # print('new_states', new_states)

    #     algo = RAOStar(geordi_model, cc=0.1, debugging=False, cc_type='o', fixed_horizon = 3)

    #     b_init = {geordi_model.current_state: 1.0}
    #     P, G = algo.search(b_init)

    #     print(P)
    #     if len(P.keys()) == 1 and m_idx == 0:
    #         print("Crashed", maneuver)
    #         crash = 1
    #         j = 0
    #         break
        
    #     if len(P.keys()) == 1:
    #         print("Turn safely")
    #         break

    # times.append((j+1)/30.0*4.8)
    # crashes.append(crash)

# crashes = np.array(crashes)
# times = np.array(times)
# print(np.sum(crashes)*1.0/len(crashes))
# print(np.mean(times))
# print(crashes)
# print(test_maneuvers)
# print(times)
