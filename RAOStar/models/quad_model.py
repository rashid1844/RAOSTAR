#!/usr/bin/env python 

# Quad Copter model to be used with RAO* 
# Yun Chang 2017 
# yunchang@mit.edu 

# The idea is to create a quadcopter model, with simplistic actions and discrete transitions 
# as to demonstrate RAO*, the belief state however, incorporates another vehicle (quadcopter, 
# tutle bot) that will also be performing actions. Collision possibility is modeled as risk. 

""" environment example 
_____________
|0,2|1,2|2,2|
|___|___|___|
|0,1|1,1|2,1|
|___|___|___|
|0,0|1,0|2,0|
|___|___|___|
"""


import numpy as np 

class QuadModel(object):
	def __init__(self, world_size, goalCoord):
		self.envSize = world_size
		# environment size (x,y)
		# model walls as risks 
		self.goal = goalCoord # goal specify direction as well as coordinate (x,y,thet)
		self.optimization = "minimize"
		self.angle_mapping = {0:[1,0],45:[1,1],90:[0,1],135:[-1,1],180:[-1,0], \
								225:[-1,-1],270:[0,-1],315:[1,-1]}
		# note state include time (depth) 
		
	def actions(self, state):
		if state[0][0] == self.goal[0] and state[0][1] == self.goal[1]: 
			return [] # at goal no more actions required 
		elif state[0][0] == 0 or state[0][1] == 0 \
					or state[0][0] == self.envSize[0]-1 or state[0][1] == self.envSize[1]-1:
			return [] # at boundary, risk is 1 and action is none 
		else: 
			return ["forward", "turn-right-45", "turn-left-45"] # was originally going to have turn right and left 90 can easily add it in 

	def is_terminal(self, state):
		return (state[0][0] == self.goal[0] and state[0][1] == self.goal[1])

	def state_transitions(self, state, action):
		# note the state is both where the controlled quad is in the env and where the other quad is in the env
		# model such that the other vehicle always move forward with probability 0.5, stay in place with probability 0.5
		# state is given as ((x1,y1,theta1),(x2,y2, theta2)) where (x1,y1,theta1) is coordinate of controlled vehicle 
		# and (x2,y2,theta2) is other 
		quadState = state[0] # controlled quad
		guestState = state[1] # uncontrolled robot 
		if action == "forward":
			direction = self.angle_mapping[quadState[2]]
			newquadState = (quadState[0] + direction[0], quadState[1] + direction[1], quadState[2], quadState[3]+1)
		elif action == "turn-right-45":
			angle = 45 # int(action[11:])
			newAng = (quadState[2] - angle) % 360 
			newquadState = (quadState[0], quadState[1], newAng, quadState[3]+1)
		elif action == "turn-left-45":
			angle = 45 # int(action[10:])
			newAng = (quadState[2] + angle) % 360 
			newquadState = (quadState[0], quadState[1], newAng, quadState[3]+1)
		else: 
			newquadState = quadState
		guestDir = self.angle_mapping[guestState[2]]
		# 50 % chance of moving forward 
		guestState = (guestState[0], guestState[1], guestState[2], guestState[3]+1) # increase time step 
		newguestState = (guestState[0] + guestDir[0], guestState[1] + guestDir[1], guestState[2], guestState[3])
		return [((newquadState,guestState),0.5), ((newquadState,newguestState),0.5)]

	def observations(self, state):
		return [(state, 1.0)] # assume observations is deterministic 

	def state_risk(self, state):
		if (state[0][0], state[0][1], state[0][3]) == (state[1][0], state[1][1], state[1][3]):
			return 1.0
		elif state[0][0] == 0 or state[0][1] == 0 \
					or state[0][0] == self.envSize[0]-1 or state[0][1] == self.envSize[1]-1:
			return 1.0
		return 0.0

	def costs(self, action):
		if action == "turn-right-45" or action == "turn-left-45":
			return 1. # bias in going forward 
		else:
			return 1.5

	def values(self, state, action):
		# return value (heuristic + cost)
		return self.costs(action) + self.heuristic(state)

	def heuristic(self, state):
		# square of euclidean distance + angle heuristic 
		disp = [(self.goal[i] - state[0][i]) for i in range(2)]
		dist = np.sqrt(sum([(self.goal[i] - state[0][i])**2 for i in range(2)]))
		# see which direction the goal is at 
		# a = state[0][2]/360*2*np.pi
		# curr_dir = [np.cos(a), np.sin(a)]
		# ang_diff = np.arccos((disp[0]*curr_dir[0] + disp[1]*curr_dir[1])/dist)
		# ang_h = ang_diff/(np.pi/4.)
		return dist #+ ang_h

	def execution_risk_heuristic(self, state):
		return 0 # don't have a good heuristic for this yet 

	