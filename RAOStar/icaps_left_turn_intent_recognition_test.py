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
dist_errors = []
n_tests = 1000
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
        best_indices = []
        best_positions = []
        for k, m in enumerate(maneuvers):
            pft = pfts[k]
            # pft.pos_sigma = pft.pos_sigma*4
            best_index, likelihood, best_pos = pft.calculateMostLikelyPointPDF(obs)
            best_indices.append(best_index)
            best_positions.append(best_pos)
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
            est_pos = best_positions[0]
            est_ind = best_indices[0]
            est_fut_pos = pfts[0].pos_mu[-1]
            est_pos_diff = [est_fut_pos[0]-est_pos[0], est_fut_pos[1]-est_pos[1]]
        else:
            est_pos = best_positions[1]
            est_ind = best_indices[1]
            est_fut_pos = pfts[1].pos_mu[-1]
            est_pos_diff = [est_fut_pos[0]-est_pos[0], est_fut_pos[1]-est_pos[1]]

        # print('-------------------')
        # print(est_pos_diff)

        current_pos = traj[j-10,:]
        future_index = j-10+24-est_ind
        if future_index >= 0 and future_index <= 24:
            future_pos = traj[future_index,:]
            pos_diff = [future_pos[0]-current_pos[0],future_pos[1]-current_pos[1]]
            # print(pos_diff)
            dist_error = np.sqrt((pos_diff[0]-est_pos_diff[0])**2 + (pos_diff[1]-est_pos_diff[1])**2)
            dist_errors.append(dist_error)

 
hits = np.array(hits)
print(hits)
print(np.sum(hits)*1.0/len(hits))

dist_errors = np.array(dist_errors)
print(dist_errors)
print(np.mean(dist_errors))