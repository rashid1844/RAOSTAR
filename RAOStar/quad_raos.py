#!/usr/bin/env python

# author: Yun Chang
# email: yunchang@mit.edu
# rao* on quadcopter with guest model

from utils import import_models
import_models()

from quad_model import QuadModel
from raostar import RAOStar

import graph_to_json

from iterative_raostar import *

#### Run RAO star on Scenario ####
# Simulation conditions 
world_size = (7,7) # note the boundaries are walls 
goal_state = (5,5,90)
quad_init = (1,1,90,0) # (x,y,theta,t)
guest_init = (3,1,90,0)

# note the boundary of the world (ex anything with row column 0 and the upper bound)
# is the wall
model = QuadModel(world_size, goal_state)
algo = RAOStar(model, cc=0.5, debugging=False)

b_init = {(quad_init, guest_init): 1.0} # initialize belief state 

P, G = algo.search(b_init)

# # get the policies that does not give none
# P_notNone = {}
# for i in P:
#     if P[i] != 'None':
#         P_notNone[i] = P[i]

# print(P_notNone)

# # print out the policy for each state of guest 
# most_likely_policy(G, model)

gshow = graph_to_json.policy_to_json(G, 0.5, 'quadraos.json')

