#!/usr/bin/env python

# author: Yun Chang
# email: yunchang@mit.edu
# rao* on fire-ice-sand model

from utils import import_models
import_models()
from fismodel import fire_ice_sand_Model
from raostar import RAOStar
from iterative_raostar import *

chance_constraint = 0.15

start = [3,0]
goal = [3,6]
model = fire_ice_sand_Model(7, [], [], [], start, goal)

algo = RAOStar(model, cc=chance_constraint, debugging=False)

b_init = {(start[0], start[1], 0): 1.0}
P, G = algo.search(b_init)

P = clean_policy(P)

model.print_model()
model.print_policy(P)

most_likely_policy(G, model)

# g = graph_to_json.policy_to_json(G, chance_constraint, "results/r2d2_raos.json")
# print(g)
