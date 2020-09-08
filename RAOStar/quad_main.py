#!/usr/bin/env python 

# Visual demo for rao* implementation on quad-with-friend model  
# Yun Chang 2017 
# yunchang@mit.edu

import quad_sim
from utils import import_models
import_models()

from quad_model import QuadModel
from raostar import RAOStar
import numpy as np 

#### Run RAO star on Scenario ####
# Simulation conditions 
world_size = (7,7) # note the boundaries are walls 
goal_state = (5,5,90)
quad_init = (1,1,90,0) # (x,y,theta,t)
guest_init = (3,1,90,0)

### Plce state string in nicer format
def get_state_from_string(state_string):
	guest_state = ''
	quad_state = ''
	storing_quad_state = 0
	storing_guest_state = 0
	for letter in state_string:
		if letter == ')':
			if storing_quad_state == 1: 
				storing_quad_state = 2
			if storing_guest_state == 1:
				storing_guest_state = 2
		if storing_quad_state == 1:
			quad_state += letter
		if storing_guest_state == 1:
			guest_state += letter
		if letter == '(':
			if storing_quad_state == 0:
				storing_quad_state = 1
			elif storing_quad_state == 2:
				storing_guest_state = 1
	quad_state = quad_state[1:]
	qs = quad_state.split(',')
	q = tuple([int(i) for i in qs])
	gs = guest_state.split(',')
	g = tuple([int(i) for i in gs])
	return (q,g)

# note the boundary of the world (ex anything with row column 0 and the upper bound)
# is the wall
model = QuadModel(world_size, goal_state)
algo = RAOStar(model, cc=0.0001)

b_init = {(quad_init, guest_init): 1.0} # initialize belief state 

P, G = algo.search(b_init)
# print(P)
sorted_policy = {} # sort policy by quad state 
for statestring in P.keys():
	state = get_state_from_string(statestring)
	if state[0] not in sorted_policy:
		sorted_policy[state[0]] = {state[1]:P[statestring]}
	else: 
		sorted_policy[state[0]][state[1]] = P[statestring]

print(sorted_policy)
S = quad_sim.Simulator(world_size[0], world_size[1], sorted_policy, model, (quad_init, guest_init))
S.draw_grid()
S.draw_quad(quad_init)
S.draw_guest(guest_init)
S.done()
